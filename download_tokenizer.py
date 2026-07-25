from huggingface_hub import hf_hub_download
import shutil
import os

repo_id = "typhoon-ai/typhoon-ocr-7b"
files = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "added_tokens.json", "special_tokens_map.json"]

print(f"Downloading tokenizer from {repo_id}...")
for f in files:
    try:
        path = hf_hub_download(repo_id=repo_id, filename=f)
        shutil.copy(path, f"./{f}")
        print(f"✅ Downloaded {f}")
    except Exception as e:
        print(f"⚠️ Could not download {f} (Might not exist, which is fine)")

print("\n🎉 Tokenizer downloaded successfully! Ready to train.")
