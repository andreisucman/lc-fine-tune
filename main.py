import os
import torch
from datasets import Dataset,load_dataset,interleave_datasets
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback
)
from trl import SFTTrainer
from unsloth.chat_templates import get_chat_template
from huggingface_hub import create_repo

# Configuration
MODEL_ID = "google/gemma-3-4b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LORA_RANK = 16  # Increased rank for better performance
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
GRAD_ACCUM_STEPS = 1
BATCH_SIZE = 8  # Increased for A100 80GB
EPOCHS = 5
LEARNING_RATE = 2e-5  # Optimized learning rate
OUTPUT_DIR = "./gemma-3-4b-it-lora-finetuned"
REPO_ID = "Sunchain/gemma-3-4b-it-merged"

# Load environment variables
HF_TOKEN = os.environ.get("HF_TOKEN")
assert HF_TOKEN, "Missing Hugging Face token in environment variable 'HF_TOKEN'"

# Configure tokenizer for Flash Attention
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    padding_side="right",  # Required for Flash Attention
    truncation_side="right",
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)

# QLoRA Configuration (Double Quantization + NF4)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True  # Double quantization for memory efficiency
)

# Load model with Flash Attention
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN,
    torch_dtype=torch.bfloat16
)

# Prepare model for QLoRA training
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False  # Required for Flash Attention
model.config.pretraining_tp = 1  # Disable tensor parallelism
model.config.attn_implementation="flash_attention_2" # Enable Flash Attention

# LoRA Configuration (covering all linear layers)
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

# ---------------------------
# Data preparation functions
# ---------------------------

cnn_dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", trust_remote_code=True, split=f"train[:3000]")
qsum_dataset = load_dataset("pszemraj/qmsum-cleaned", None, split = "train[:3000]")
lex_dataset = load_dataset("CJWeiss/LexSumm", "multishort", split = "train[:3000]")

cnn_list = [dict(item) for item in cnn_dataset]
qsum_list = [dict(item) for item in qsum_dataset]
lex_list = [dict(item) for item in lex_dataset]

def format_conversations(data_list, type):
    items = []

    for item in data_list:
        if type == "cnn":
            user_content = f"{str(item.get('article', ''))}".strip()
            assistant_content = str(item.get('highlights', '')).strip()
        elif type == "qsum":
            user_content = f"{str(item.get('input', ''))}".strip()
            assistant_content = str(item.get('output', '')).strip()
        elif type == "lex":
            user_content = f"{str(item.get('input', ''))}".strip()
            assistant_content = str(item.get('output', '')).strip()
        else:
            continue

        items.append({"conversations":[{"role": "user", "content": user_content},{"role": "assistant", "content": assistant_content}]})

    return items

cnn_dataset = format_conversations(cnn_dataset, "cnn")
qsum_dataset = format_conversations(qsum_dataset, "qsum")
lex_dataset = format_conversations(lex_dataset, "lex")

cnn_dataset = Dataset.from_list(cnn_dataset).shuffle()
qsum_dataset = Dataset.from_list(qsum_dataset).shuffle()
lex_dataset = Dataset.from_list(lex_dataset).shuffle()

cnn_eval = cnn_dataset.select(range(2700, 3000)) 
qsum_eval = qsum_dataset.select(range(2700, 3000))
lex_eval = lex_dataset.select(range(2700, 3000)) 

cnn_train = cnn_dataset.select(range(0, 2700))
qsum_train = qsum_dataset.select(range(0, 2700))
lex_train = lex_dataset.select(range(0, 2700))

print("datasets defined")

train_dataset = interleave_datasets(
    [cnn_train, qsum_train, lex_train],
    probabilities=[0.25,0.25,0.5],
    seed=42
)

eval_dataset = interleave_datasets(
    [cnn_train, qsum_train, lex_train],
    probabilities=[0.25,0.25,0.5],
    seed=42
)

print("interleaving done")

def formatting_prompts_func(examples):
   convos = examples["conversations"]
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
   return { "text" : texts, }

train_dataset = train_dataset.map(formatting_prompts_func, batched = True)
eval_dataset = eval_dataset.map(formatting_prompts_func, batched = True)

print("datasets ready")

del cnn_dataset 
del qsum_dataset
del lex_dataset

del cnn_eval
del qsum_eval 
del lex_eval 

del cnn_train
del qsum_train 
del lex_train 

# ---------------------------
# Training setup with Flash Attention optimizations
# ---------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    optim="paged_adamw_8bit", 
    weight_decay = 0.01,
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
    bf16=True,  # Force BF16 for A100
    fp16=False,  # Disable FP16 when using BF16
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to="none",
    seed = 3407,
    gradient_checkpointing=True,  # Enable for memory savings
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_steps=-1,
    group_by_length=True,  # Improves efficiency with packing
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

if os.path.exists(OUTPUT_DIR) and any(os.scandir(OUTPUT_DIR)):
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()

# Save final model
trainer.save_model(OUTPUT_DIR)

create_repo(REPO_ID, token=HF_TOKEN, private=True, exist_ok=True)

# Push to Hub (adapter only)
trainer.model.push_to_hub(
    REPO_ID,
    use_temp_dir=False,
    token=HF_TOKEN,
    private=True
)

tokenizer.push_to_hub(
    REPO_ID,
    use_temp_dir=False,
    token=HF_TOKEN
)

print("Model pushed to Hub!")