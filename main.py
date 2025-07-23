import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer
from unsloth.chat_templates import get_chat_template
from huggingface_hub import create_repo
from accelerate import Accelerator

# === CONFIGURATION ===
MODEL_ID = "google/gemma-3-4b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
GRAD_ACCUM_STEPS = 1
BATCH_SIZE = 4
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
print("📦 Loading LexSumm dataset...")
lex_dataset = load_dataset("CJWeiss/LexSumm", "multishort", split="train")

def format_lexsum(example):
    user = example.get("input", "").strip()
    assistant = example.get("output", "").strip()
    return {
        "conversations": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant}
        ]
    }

print("🛠️ Formatting conversations...")
lex_dataset = lex_dataset.map(format_lexsum, num_proc=4, remove_columns=lex_dataset.column_names)

# === Tokenize with chat template ===
def preprocess_function(example):
    formatted = tokenizer.apply_chat_template(example["conversations"], tokenize=False, add_generation_prompt=False)
    if not formatted.strip():
        return {"input_ids": [], "attention_mask": []}
    return tokenizer(formatted, truncation=True, padding="max_length")

accelerator = Accelerator()

with accelerator.main_process_first():
    tokenized_dataset = lex_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_unused_columns=True,
        desc="Tokenizing dataset"
    )

# === Filter empty examples ===
tokenized_dataset = tokenized_dataset.filter(
    lambda example: len(example["input_ids"]) > 0 and any(id != tokenizer.pad_token_id for id in example["input_ids"]),
    num_proc=4
)

# === Split train/validation ===
split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

# === Collator ===
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

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
    logging_dir=os.path.join(FINETUNE_DIR, "logs"),
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
    disable_tqdm=False,
    logging_first_step=True,
)

# === TRAINING ===
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

if os.path.exists(FINETUNE_DIR) and any(os.scandir(FINETUNE_DIR)):
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()

# === SAVE LoRA MODEL AND PUSH TO HUB ===
trainer.save_model(FINETUNE_DIR)
create_repo(ADAPTER_REPO_ID, token=HF_TOKEN, private=True, exist_ok=True)

trainer.model.push_to_hub(ADAPTER_REPO_ID, token=HF_TOKEN, private=True)
tokenizer.push_to_hub(ADAPTER_REPO_ID, token=HF_TOKEN, private=True)

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
