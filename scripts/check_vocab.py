import json
import os

def check_vocab_size(file_path="tokenizer.json"):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found!")
        return
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        # In most tokenizer.json files, vocab is in model -> vocab
        vocab = data.get("model", {}).get("vocab", {})
        actual_size = len(vocab)
        print(f"Actual Vocabulary Size from {file_path}: {actual_size:,}")
        
        # Round up to nearest multiple of 64 for GPU efficiency
        suggested_config_size = ((actual_size + 63) // 64) * 64
        print(f"Suggested VOCAB_SIZE for config.py: {suggested_config_size}")
        
        return suggested_config_size

if __name__ == "__main__":
    check_vocab_size()