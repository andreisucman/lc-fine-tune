import os
import torch
from datasets import load_dataset,interleave_datasets
import evaluate
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
import numpy as np

# Configuration
MODEL_ID = "google/gemma-3-4b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LORA_RANK = 8  # Increased rank for better performance
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
GRAD_ACCUM_STEPS = 1
BATCH_SIZE = 8  # Increased for A100 80GB
EPOCHS = 5
LEARNING_RATE = 1e-5  # Optimized learning rate
OUTPUT_DIR = "./gemma-3-4b-it-lora-finetuned"
REPO_ID = "Sunchain/gemma-3-4b-it-brevity"

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

SUMMARY_TASKS = [
    {"name": "abisee/cnn_dailymail", "config": "3.0.0", "input_field": "article", "summary_field": "highlights", "size": 30000},
    {"name": "pszemraj/qmsum-cleaned", "config": None, "input_field": "input", "summary_field": "output", "size": 15000},
    {"name": "CJWeiss/LexSumm", "config": "eurlexsum", "input_field": "input", "summary_field": "output", "size": 30000},
]

def filter_by_length(example, max_summary_ratio=0.3, min_input_words=100):
    input_len = len(example["input"].split())
    output_len = len(example["output"].split())
    return input_len >= min_input_words and output_len / input_len <= max_summary_ratio


def load_and_format(dataset_name, input_field, summary_field, max_items, config=None):
    print(f"📥 Loading {dataset_name} ({max_items} samples)")
    if config:
        raw = load_dataset(dataset_name, config, trust_remote_code=True, split=f"train[:{max_items}]")
    else:
        raw = load_dataset(dataset_name, trust_remote_code=True, split=f"train[:{max_items}]")

    raw = raw.rename_columns({input_field: "input", summary_field: "output"})

    # Remove non-uniform fields like "id" to prevent schema mismatch
    keep_cols = {"input", "output"}
    drop_cols = [col for col in raw.column_names if col not in keep_cols]
    if drop_cols:
        raw = raw.remove_columns(drop_cols)

    def to_chat_format(example):
        instruction = "Summarize the following text:"
        input_text = example["input"].strip()
        output_text = example["output"].strip()
        chat = [
            {"role": "user", "content": f"{instruction}\n{input_text}"},
            {"role": "assistant", "content": output_text}
        ]
        return {"conversations": chat}

    dataset = raw.map(to_chat_format)
    dataset = dataset.filter(filter_by_length)
    return dataset.shuffle(seed=42)


# Load all datasets individually
formatted_datasets = [
    load_and_format(task["name"], task["input_field"], task["summary_field"], task["size"], task.get("config"))
    for task in SUMMARY_TASKS
]

# Interleave datasets with roughly equal probability
interleaved_dataset = interleave_datasets(formatted_datasets, seed=42)

# Split into train/eval (90%/10%)
total_len = len(interleaved_dataset)
train_len = int(0.9 * total_len)
train_dataset = interleaved_dataset.select(range(train_len))
eval_dataset = interleaved_dataset.select(range(train_len, total_len))

print(f"✅ Total examples: {total_len}, train: {train_len}, eval: {total_len - train_len}")

# Format for Gemma chat style
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False).removeprefix("<bos>")
        for convo in convos
    ]
    return {"text": texts}

train_dataset = train_dataset.map(formatting_prompts_func, batched=True, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True, remove_columns=eval_dataset.column_names)

print("datasets ready")

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Decode token ids to strings (assuming tokenizer is available)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some labels might be padded with -100 (ignore index), replace with pad token id or empty string
    decoded_labels = [label.strip() for label in decoded_labels]

    # Compute ROUGE scores
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    # Extract and format main ROUGE scores (rouge1, rouge2, rougeL)
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # Optionally add mean prediction length for logging
    prediction_lens = [len(pred.split()) for pred in decoded_preds]
    result["gen_len"] = np.mean(prediction_lens)

    return result

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
    metric_for_best_model="rouge1",
    greater_is_better=True,
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
    compute_metrics=compute_metrics,
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
    private=True,
)

tokenizer.push_to_hub(
    REPO_ID,
    use_temp_dir=False,
    token=HF_TOKEN
)

print("Model pushed to Hub")