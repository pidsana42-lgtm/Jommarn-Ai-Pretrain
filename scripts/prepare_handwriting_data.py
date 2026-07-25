import os
import torch
from datasets import load_dataset
from PIL import Image
from torchvision import transforms
from transformers import PreTrainedTokenizerFast

class ThaiHandwritingDataset(torch.utils.data.Dataset):
    """
    Custom Dataset for Thai Handwriting OCR.
    Bridges handwritten images with Jommarn-Omni's Thinker.
    """
    def __init__(self, split="train", tokenizer_path="tokenizer.json", img_size=224):
        print(f"Loading {split} split from iapp/thai_handwriting_dataset...")
        self.dataset = load_dataset("iapp/thai_handwriting_dataset", split=split)
        
        # Load the Gemma-4 Tokenizer
        if os.path.exists(tokenizer_path):
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        else:
            raise FileNotFoundError("Please run scripts/download_tokenizer.py first!")
            
        self.tokenizer.pad_token = "<|pad|>"
        
        # Standard Vision transformations for Jommarn-Vision
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        text = item["text"]
        
        # Process image
        pixel_values = self.transform(image)
        
        # Process text (Tokenization)
        # We append <|endoftext|> to help the model learn when to stop
        tokens = self.tokenizer(
            text + "<|endoftext|>", 
            truncation=True, 
            max_length=128, 
            padding="max_length", 
            return_tensors="pt"
        )
        
        return {
            "images": pixel_values,
            "input_ids": tokens["input_ids"].squeeze(0),
            "targets": tokens["input_ids"].squeeze(0)
        }

def download_and_preview():
    """
    Downloads the dataset and prints a sample to verify.
    """
    try:
        # Check if tokenizer exists
        if not os.path.exists("tokenizer.json"):
            print("Downloading tokenizer first...")
            from scripts.download_tokenizer import download_gemma_tokenizer
            download_gemma_tokenizer()

        # Initialize Dataset
        thai_ds = ThaiHandwritingDataset(split="train")
        print(f"Successfully loaded {len(thai_ds)} handwriting samples!")
        
        # Preview a sample
        sample = thai_ds[0]
        print(f"Image tensor shape: {sample['images'].shape}")
        print(f"Target tokens sample: {sample['input_ids'][:10]}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Tip: You might need to install datasets: pip install datasets")

if __name__ == "__main__":
    download_and_preview()