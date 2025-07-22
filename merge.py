from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from peft import PeftModel
import os

# === CONFIGURATION ===
BASE_MODEL_ID = "google/gemma-3-4b-it"
ADAPTER_REPO = "Sunchain/gemma-3-4b-it-dolly-alpaca-ro"
MERGED_REPO_ID = "Sunchain/gemma-3-4b-it-merged"
MERGED_DIR = "./merged_model"
HF_TOKEN = os.environ.get("HF_TOKEN")
assert HF_TOKEN, "Missing HF_TOKEN in environment"

# === LOAD BASE MODEL AND ADAPTER ===
print("🔄 Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)

print("🔄 Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_REPO)

# === MERGE LORA INTO BASE MODEL ===
print("🔀 Merging LoRA into base model...")
merged_model = model.merge_and_unload()

# === SAVE MERGED MODEL (PyTorch format to avoid shared tensor issue) ===
print(f"💾 Saving merged model to: {MERGED_DIR}")
merged_model.save_pretrained(MERGED_DIR, safe_serialization=False)

# === SAVE TOKENIZER from adapter repo ===
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_REPO)
tokenizer.save_pretrained(MERGED_DIR)

# === LOAD AND SAVE PREPROCESSOR from base model repo ===
try:
    print("🔄 Loading preprocessor from base model repo...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
    processor.save_pretrained(MERGED_DIR)
except Exception as e:
    print(f"⚠️ No preprocessor found or failed to load: {e}")

# === PUSH TO HUB (disable safetensors to avoid issue) ===
print(f"🚀 Pushing merged model to: https://huggingface.co/{MERGED_REPO_ID}")
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
    pass  # silently skip if processor does not exist or fails

print("✅ Merge and push complete.")
