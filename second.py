import os
from huggingface_hub import login
import torch
import pandas as pd
import requests
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig,PeftModel

# Login into Hugging Face Hub
HF_TOKEN = os.environ.get("HF_TOKEN")# If you are running inside a Google Colab
login(HF_TOKEN)

OUTPUT_DIR = "./gemma-3-4b-pt-lora-finetuned"
REPO_ID = "Sunchain/gemma-3-4b-pt-dolly-alpaca-ro"

# ---------------------------
# Data preparation functions
# ---------------------------

def download_file(url, output_path):
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    if not os.path.exists(output_path):
        response = requests.get(url)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(response.content)

def extract_prompt_response(row):
    prompt = row["instruction"].strip()
    if row.get("context"):
        prompt += " " + str(row["context"]).strip()
    if row.get("input"):
        prompt += " " + str(row["input"]).strip()

    response = ""
    if "response" in row and pd.notna(row["response"]):
        response = str(row["response"]).strip()
    elif "output" in row and pd.notna(row["output"]):
        response = str(row["output"]).strip()

    return {"prompt": prompt, "response": response}


# Hugging Face model id
model_id = "google/gemma-3-4b-pt" # or `google/gemma-3-4b-pt`, `google/gemma-3-12b-pt`, `google/gemma-3-27b-pt`
model_class = AutoModelForCausalLM
# Select model class based on id

# Check if GPU benefits from bfloat16
if torch.cuda.get_device_capability()[0] >= 8:
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float16

# Define model init arguments
model_kwargs = dict(
    attn_implementation="flash_attention_2", # Use "flash_attention_2" when running on Ampere or newer GPU
    torch_dtype=torch_dtype, # What torch dtype to use, defaults to auto
    device_map="auto", # Let torch decide how to load the model
)

# BitsAndBytesConfig: Enables 4-bit quantization to reduce model size/memory usage
model_kwargs["quantization_config"] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=model_kwargs['torch_dtype'],
    bnb_4bit_quant_storage=model_kwargs['torch_dtype'],
)

# Load model and tokenizer
model = model_class.from_pretrained(model_id, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it") 

# ---------------------------
# Download and process datasets
# ---------------------------

# Dolly
dolly_path = "data/parquet/dolly/train.parquet"
download_file(
    "https://huggingface.co/api/datasets/databricks/databricks-dolly-15k/parquet/default/train/0.parquet",
    dolly_path
)
dolly_df = pd.read_parquet(dolly_path)
dolly_data = dolly_df.apply(extract_prompt_response, axis=1).tolist()

# Alpaca
alpaca_path = "data/parquet/alpaca/train.parquet"
download_file(
    "https://huggingface.co/api/datasets/yahma/alpaca-cleaned/parquet/default/train/0.parquet",
    alpaca_path
)
alpaca_df = pd.read_parquet(alpaca_path)
alpaca_data = alpaca_df.apply(extract_prompt_response, axis=1).tolist()

# Legal Summarization
legal_path = "legal-summarization-ro.jsonl"
download_file(
    "https://lxi-data.fra1.digitaloceanspaces.com/summarize-legal-ro-18k.jsonl",
    legal_path
)
legal_data = []
with open(legal_path) as f:
    for line in f:
        example = json.loads(line)
        legal_data.append({
            "prompt": example["instruction"],
            "response": example["response"]
        })

# Combine datasets
combined_data = dolly_data + alpaca_data + legal_data
print(f"Total training examples: {len(combined_data)}")

# Convert to HF dataset
dataset = Dataset.from_list(combined_data).train_test_split(test_size=0.1)

# Format for chat template
def format_chat(example):
    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["response"]}
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

dataset = dataset.map(format_chat, remove_columns=["prompt", "response"])

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head", "embed_tokens"] # make sure to save the lm_head and embed_tokens as you train the special tokens
)

args = SFTConfig(
    output_dir=OUTPUT_DIR,         # directory to save and repository id
    max_seq_length=1024,                     # max sequence length for model and packing of the dataset
    packing=True,                           # Groups multiple samples in the dataset into a single sequence
    num_train_epochs=8,                     # number of training epochs
    per_device_train_batch_size=4,          # batch size per device during training
    gradient_accumulation_steps=8,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=50,                       # log every 10 steps
    save_strategy="epoch",                  # save checkpoint every epoch
    learning_rate=2e-5,                     # learning rate, based on QLoRA paper
    fp16=True if torch_dtype == torch.float16 else False,   # use float16 precision
    bf16=True if torch_dtype == torch.bfloat16 else False,   # use bfloat16 precision
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    push_to_hub=True,                       # push model to hub
    report_to="tensorboard",                # report metrics to tensorboard
    dataset_kwargs={
        "add_special_tokens": False, # We template with special tokens
        "append_concat_token": True, # Add EOS token as separator token between examples
    }
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    peft_config=peft_config,
    processing_class=tokenizer
)

# Start training, the model will be automatically saved to the Hub and the output directory
trainer.train()

# Save the final model again to the Hugging Face Hub
trainer.save_model()

# free the memory again
del model
del trainer
torch.cuda.empty_cache()

# Load Model base model
model = model_class.from_pretrained(model_id, low_cpu_mem_usage=True)

# Merge LoRA and base model and save
peft_model = PeftModel.from_pretrained(model, args.output_dir)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained(OUTPUT_DIR, safe_serialization=True, max_shard_size="2GB")

processor = AutoTokenizer.from_pretrained(args.output_dir)
processor.save_pretrained(OUTPUT_DIR)

merged_model.push_to_hub(
    REPO_ID,
    use_temp_dir=False,
    token=os.environ["HF_TOKEN"]
)
tokenizer.push_to_hub(
    REPO_ID,
    use_temp_dir=False,
    token=os.environ["HF_TOKEN"]
)

print("Training complete and model pushed to Hub!")