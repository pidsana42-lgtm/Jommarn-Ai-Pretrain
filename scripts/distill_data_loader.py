import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

class DistilledLogitsDataset(Dataset):
    """
    Dataset for Pre-training with Teacher Logits (Logits Distillation).
    Loads distilled dataset from Hugging Face Hub (e.g. Jommarn/qwen2.5-7b-pretrain-distilled-logits)
    which contains:
      - text: Raw text
      - top10_token_ids: List of top 10 teacher token IDs per sequence position
      - top10_logits: List of top 10 teacher logit floats per sequence position
    """
    def __init__(self, dataset_name="Jommarn/qwen2.5-7b-pretrain-distilled-logits", tokenizer_path="tokenizer.json", context_length=512, split="train"):
        print(f"📦 Loading Distilled Logits Dataset from Hugging Face: {dataset_name} ({split})...")
        self.context_length = context_length
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path) if os.path.exists(tokenizer_path) else None
        
        try:
            self.raw_dataset = load_dataset(dataset_name, split=split)
        except Exception as e:
            print(f"⚠️ Warning: Failed to load from HF Hub ({e}). Using empty dataset...")
            self.raw_dataset = []

        # Prepare 1024 / 512-token chunks with matching teacher logits
        self.samples = []
        target_len = context_length + 1
        
        for item in self.raw_dataset:
            text = item.get("text", "")
            top10_logits = item.get("top10_logits") or item.get("top_logits")
            top10_ids = item.get("top10_token_ids") or item.get("top_token_ids")
            
            if not top10_logits or not top10_ids:
                # Tokenize fallback
                if self.tokenizer and text:
                    tokens = self.tokenizer.encode(text)
                    top10_logits = [[0.0]*10 for _ in tokens]
                    top10_ids = [[t]*10 for t in tokens]
                else:
                    continue
            else:
                if self.tokenizer and text:
                    tokens = self.tokenizer.encode(text)
                else:
                    tokens = [row[0] for row in top10_ids]
                    
            tokens_t = torch.tensor(tokens, dtype=torch.long)
            top10_logits_t = torch.tensor(top10_logits, dtype=torch.float32)
            top10_ids_t = torch.tensor(top10_ids, dtype=torch.long)
            
            seq_len = min(tokens_t.size(0), top10_logits_t.size(0))
            
            # Slice into chunks of target_len
            for start in range(0, seq_len - target_len + 1, context_length):
                chunk_tokens = tokens_t[start:start+target_len]
                chunk_logits = top10_logits_t[start:start+target_len]
                chunk_ids = top10_ids_t[start:start+target_len]
                self.samples.append((chunk_tokens, chunk_logits, chunk_ids))
                
        print(f"✅ Prepared {len(self.samples)} sequence chunks (Length={context_length}) for Logits Distillation!")

    def __len__(self):
        return len(self.samples) if len(self.samples) > 0 else 100

    def __getitem__(self, idx):
        if len(self.samples) == 0:
            # Fallback dummy tensors
            target_len = self.context_length + 1
            tokens = torch.randint(0, 152064, (target_len,))
            teacher_logits = torch.randn(target_len, 10)
            teacher_top_ids = torch.randint(0, 152064, (target_len, 10))
            return tokens[:-1], tokens[1:], teacher_logits[:-1], teacher_top_ids[:-1]
            
        chunk_tokens, chunk_logits, chunk_ids = self.samples[idx % len(self.samples)]
        
        xb = chunk_tokens[:-1]
        yb = chunk_tokens[1:]
        teacher_logits = chunk_logits[:-1]
        teacher_top_ids = chunk_ids[:-1]
        
        return xb, yb, teacher_logits, teacher_top_ids

def get_distill_loader(dataset_name="Jommarn/qwen2.5-7b-pretrain-distilled-logits", batch_size=4, context_length=512, tokenizer_path="tokenizer.json"):
    dataset = DistilledLogitsDataset(dataset_name=dataset_name, tokenizer_path=tokenizer_path, context_length=context_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

if __name__ == '__main__':
    loader = get_distill_loader(batch_size=2, context_length=128)
    for xb, yb, t_logits, t_ids in loader:
        print("xb shape:", xb.shape)
        print("yb shape:", yb.shape)
        print("teacher logits shape:", t_logits.shape)
        print("teacher top ids shape:", t_ids.shape)
        break
