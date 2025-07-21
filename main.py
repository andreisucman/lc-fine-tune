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

# Configuration
MODEL_ID = "google/gemma-3-4b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LORA_RANK = 8  # Increased rank for better performance
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
GRAD_ACCUM_STEPS = 1
BATCH_SIZE = 8  # Increased for A100 80GB
EPOCHS = 4
LEARNING_RATE = 1e-4  # Optimized learning rate
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

dolly_dataset = format_conversations(dolly_dataset, "dolly")
alpaca_dataset = format_conversations(alpaca_dataset, "alpaca")
legal_dataset = format_conversations(legal_dataset, "legal")

dolly_dataset = Dataset.from_list(dolly_dataset).shuffle()
alpaca_dataset = Dataset.from_list(alpaca_dataset).shuffle()
legal_dataset = Dataset.from_list(legal_dataset).shuffle()

dolly_eval = dolly_dataset.select(range(13500, 15000))  # Last 1.5k
alpaca_eval = alpaca_dataset.select(range(13500, 15000))
legal_eval = legal_dataset.select(range(44000, 48000))  # Last 4k for validation

dolly_train = dolly_dataset.select(range(0, 13500))
alpaca_train = alpaca_dataset.select(range(0, 13500))
legal_train = legal_dataset.select(range(0, 44000))

print("datasets defined")

train_dataset = interleave_datasets(
    [dolly_train, alpaca_train, legal_train],
    probabilities=[0.15, 0.15,0.70],
    seed=42
)

eval_dataset = interleave_datasets(
    [dolly_eval, alpaca_eval, legal_eval],
    probabilities=[0.15, 0.15,0.70],
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

del dolly_dataset 
del alpaca_dataset
del legal_dataset

del dolly_eval # Last 1.5k
del alpaca_eval 
del legal_eval  # Last 4k for validation

del dolly_train
del alpaca_train 
del legal_train 

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
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
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

print("trainer patching started")

from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)

print("trainer patching ended")
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