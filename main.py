import os
import torch
from datasets import Dataset, load_dataset, interleave_datasets
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback
)
from trl import SFTTrainer
from unsloth.chat_templates import get_chat_template
from huggingface_hub import create_repo

# === CONFIGURATION ===
MODEL_ID = "google/gemma-3-4b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
GRAD_ACCUM_STEPS = 1
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 2e-5
FINETUNE_DIR = "./gemma-3-4b-it-lora-finetuned"
MERGED_DIR = "./merged_model"
ADAPTER_REPO_ID = "Sunchain/gemma-3-4b-it-lora-adapter"
MERGED_REPO_ID = "Sunchain/gemma-3-4b-it-merged"

# === ENVIRONMENT ===
HF_TOKEN = os.environ.get("HF_TOKEN")
assert HF_TOKEN, "Missing Hugging Face token in environment variable 'HF_TOKEN'"

# === TOKENIZER SETUP ===
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    padding_side="right",
    truncation_side="right",
)
tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

# === QUANTIZATION CONFIG ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# === LOAD BASE MODEL ===
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN,
    torch_dtype=torch.bfloat16
)
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False
model.config.pretraining_tp = 1
model.config.attn_implementation = "flash_attention_2"

# === PEFT CONFIG ===
peft_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=LORA_DROPOUT,
    task_type="CAUSAL_LM",
    bias="none"
)

# === LOAD & FORMAT DATA ===
cnn_dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", trust_remote_code=True, split="train[:3000]")
qsum_dataset = load_dataset("pszemraj/qmsum-cleaned", split="train[:3000]")
lex_dataset = load_dataset("CJWeiss/LexSumm", "multishort", split="train")

def format_conversations(data_list, type):
    items = []
    for item in data_list:
        if type == "cnn":
            user_content = str(item.get("article", "")).strip()
            assistant_content = str(item.get("highlights", "")).strip()
        else:
            user_content = str(item.get("input", "")).strip()
            assistant_content = str(item.get("output", "")).strip()
        items.append({"conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]})
    return items

cnn_dataset = Dataset.from_list(format_conversations(cnn_dataset, "cnn")).shuffle()
qsum_dataset = Dataset.from_list(format_conversations(qsum_dataset, "qsum")).shuffle()
lex_dataset = Dataset.from_list(format_conversations(lex_dataset, "lex")).shuffle()

def split_dataset_fraction(dataset, fraction):
    if not 0 < fraction < 1:
        raise ValueError("Fraction must be between 0 and 1 (exclusive).")
    
    split_index = round(fraction * len(dataset))
    first_part = dataset.select(range(split_index))
    remaining_part = dataset.select(range(split_index, len(dataset)))

    return first_part, remaining_part

# Split datasets
cnn_train, cnn_eval = split_dataset_fraction(cnn_dataset, 0.9)
qsum_train, qsum_eval = split_dataset_fraction(qsum_dataset, 0.9)
lex_train, lex_eval = split_dataset_fraction(lex_dataset, 0.9)

# Interleaved datasets
train_dataset = interleave_datasets(
    [cnn_train, qsum_train, lex_train],
    probabilities=[0.125, 0.125, 0.75],
    seed=42
)

eval_dataset = interleave_datasets(
    [cnn_eval, qsum_eval, lex_eval],
    probabilities=[0.125, 0.125, 0.75],
    seed=42
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False).removeprefix("<bos>") for convo in convos]
    return {"text": texts}

train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)

# === TRAINING ARGS ===
training_args = TrainingArguments(
    output_dir=FINETUNE_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    optim="paged_adamw_8bit",
    weight_decay=0.01,
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
    bf16=True,
    fp16=False,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to="none",
    seed=3407,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_steps=-1,
    group_by_length=True,
)

# === TRAINING ===
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

if os.path.exists(FINETUNE_DIR) and any(os.scandir(FINETUNE_DIR)):
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()

# === SAVE LoRA MODEL AND PUSH TO HUB ===
trainer.save_model(FINETUNE_DIR)
create_repo(ADAPTER_REPO_ID, token=HF_TOKEN, private=True, exist_ok=True)

trainer.model.push_to_hub(
    ADAPTER_REPO_ID,
    token=HF_TOKEN,
    private=True
)
tokenizer.push_to_hub(
    ADAPTER_REPO_ID,
    token=HF_TOKEN,
    private=True
)

# === MERGE AND PUSH FULL MODEL ===
print("🔄 Reloading base model for merging...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

print("🔄 Loading trained LoRA adapter from output directory...")
model = PeftModel.from_pretrained(base_model, FINETUNE_DIR)

print("🔀 Merging LoRA into base model...")
merged_model = model.merge_and_unload()

print(f"💾 Saving merged model to: {MERGED_DIR}")
merged_model.save_pretrained(MERGED_DIR, safe_serialization=False)
tokenizer.save_pretrained(MERGED_DIR)

try:
    print("🔄 Saving processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.save_pretrained(MERGED_DIR)
except Exception as e:
    print(f"⚠️ No processor found or failed to load: {e}")

print(f"🚀 Pushing merged model to: https://huggingface.co/{MERGED_REPO_ID}")
create_repo(MERGED_REPO_ID, token=HF_TOKEN, private=True, exist_ok=True)
merged_model.push_to_hub(
    MERGED_REPO_ID,
    token=HF_TOKEN,
    private=True,
    safe_serialization=False
)
tokenizer.push_to_hub(
    MERGED_REPO_ID,
    token=HF_TOKEN,
    private=True
)

try:
    processor.push_to_hub(MERGED_REPO_ID, token=HF_TOKEN)
except Exception:
    pass

print("✅ Merged model pushed to Hub successfully.")
