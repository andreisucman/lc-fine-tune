from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfApi, HfFolder
import os

# Constants
OUTPUT_DIR = "./gemma-3-4b-it-lora-finetuned"
REPO_ID = "Sunchain/gemma-3-4b-it-dolly-alpaca-ro"
HF_TOKEN = os.environ.get("HF_TOKEN")
assert HF_TOKEN, "Missing Hugging Face token in environment variable 'HF_TOKEN'"

# Login (optional if token is already in env)
HfFolder.save_token(HF_TOKEN)

# Load model and tokenizer from OUTPUT_DIR
model = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR)
tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)

# Push model and tokenizer
model.push_to_hub(REPO_ID, token=HF_TOKEN, private=True)
tokenizer.push_to_hub(REPO_ID, token=HF_TOKEN, private=True)

print(f"Model and tokenizer pushed to https://huggingface.co/{REPO_ID}")
