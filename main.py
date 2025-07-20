import os
import torch
from datasets import Dataset,load_dataset,interleave_datasets
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
from unsloth.chat_templates import get_chat_template

# Configuration
MODEL_ID = "google/gemma-3-4b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LORA_RANK = 8  # Increased rank for better performance
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
GRAD_ACCUM_STEPS = 1
BATCH_SIZE = 8  # Increased for A100 80GB
MAX_SEQ_LENGTH = 128_000  # Full context length
EPOCHS = 8
LEARNING_RATE = 2e-5  # Optimized learning rate
OUTPUT_DIR = "./gemma-3-4b-it-lora-finetuned"
REPO_ID = "Sunchain/gemma-3-4b-it-dolly-alpaca-ro"

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

dolly_dataset = load_dataset("databricks/databricks-dolly-15k", split = "train")
alpaca_dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
legal_dataset = load_dataset("Sunchain/l-r-48k", split = "train")

dolly_list = [dict(item) for item in dolly_dataset]
alpaca_list = [dict(item) for item in alpaca_dataset]
legal_list = [dict(item) for item in legal_dataset]

def format_conversations(data_list, type):
    items = []

    for item in data_list:
        if type == "dolly":
            user_content = f"[EN] {str(item.get('instruction', ''))}\n{str(item.get('context', ''))}".strip()
            assistant_content = str(item.get('response', '')).strip()
        elif type == "alpaca":
            user_content = f"[EN] {str(item.get('instruction', ''))}\n{str(item.get('input', ''))}".strip()
            assistant_content = str(item.get('output', '')).strip()
        elif type == "legal":
            user_content = f"[RO] {str(item.get('instruction'))}\n{str(item.get('context', '')).strip()}".strip()
            assistant_content = str(item.get('response', '')).strip()
        else:
            continue

        items.append({"conversations":[{"role": "user", "content": user_content},{"role": "assistant", "content": assistant_content}]})

    return items

dolly_dataset = format_conversations(dolly_list, "dolly")
alpaca_dataset = format_conversations(alpaca_list, "alpaca")
legal_dataset = format_conversations(legal_list, "legal")

del alpaca_list
del dolly_list
del legal_list

dolly_dataset = Dataset.from_list(dolly_dataset).shuffle()
alpaca_dataset = Dataset.from_list(alpaca_dataset).shuffle()
legal_dataset = Dataset.from_list(legal_dataset).shuffle()

dolly_train = dolly_dataset.select(range(0, 11500))
alpaca_train = alpaca_dataset.select(range(0, 11500))
legal_train = legal_dataset.select(range(0, 44000))

del dolly_dataset
del alpaca_dataset
del legal_dataset

dataset = interleave_datasets(
    [dolly_train, alpaca_train, legal_train],
    probabilities=[0.15, 0.15,0.70],
    seed=42
)

del dolly_train
del alpaca_train
del legal_train


def formatting_prompts_func(examples):
   convos = examples["conversations"]
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
   return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)

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
    optim="paged_adamw_8bit",  # Memory-efficient optimizer
    logging_steps=10,
    bf16=True,  # Force BF16 for A100
    fp16=False,  # Disable FP16 when using BF16
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="linear",
    report_to="none",
    gradient_checkpointing=True,  # Enable for memory savings
    max_steps=-1,
    group_by_length=True,  # Improves efficiency with packing
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=MAX_SEQ_LENGTH
)

# Train
trainer.train()

# Save final model
trainer.save_model(OUTPUT_DIR)

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