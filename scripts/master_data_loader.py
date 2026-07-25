import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from datasets import load_dataset
from PIL import Image
from torchvision import transforms
from transformers import PreTrainedTokenizerFast
import glob
import json
import random
import numpy as np
from torch.utils.data.sampler import Sampler

class HugeWeightedRandomSampler(Sampler):
    """
    A custom sampler that bypasses torch.multinomial's 2^24 categories limit.
    It works by first multinomial-sampling the dataset indices based on weights,
    then randomly sampling within each dataset using numpy (which is very fast).
    """
    def __init__(self, dataset_lengths, dataset_weights, num_samples):
        self.dataset_lengths = dataset_lengths
        self.dataset_weights = dataset_weights
        self.num_samples = num_samples
        
        total_weight = sum(self.dataset_weights)
        self.probs = [w / total_weight for w in self.dataset_weights]
        self.cumulative_lengths = [0] + list(np.cumsum(dataset_lengths))
        
    def __iter__(self):
        # Sample dataset IDs based on probabilities
        dataset_choices = np.random.choice(
            len(self.dataset_lengths), 
            size=self.num_samples, 
            p=self.probs
        )
        
        # Count occurrences for each dataset to batch sample
        counts = np.bincount(dataset_choices, minlength=len(self.dataset_lengths))
        
        indices = []
        for i, count in enumerate(counts):
            if count > 0 and self.dataset_lengths[i] > 0:
                local_indices = np.random.randint(0, self.dataset_lengths[i], size=count)
                global_indices = local_indices + self.cumulative_lengths[i]
                indices.append(global_indices)
                
        all_indices = np.concatenate(indices)
        np.random.shuffle(all_indices)
        
        return iter(all_indices.tolist())
        
    def __len__(self):
        return self.num_samples

class JommarnMasterDataset(Dataset):
    """
    Unified Dataset for Jommarn-Omni.
    Handles:
    1. Thai Wiki (Text only)
    2. Thai Handwriting (Vision + Text)
    3. Appen Thai Document OCR (Vision + Text)
    """
    def __init__(self, tokenizer_path="tokenizer.json", img_size=512, mode="multimodal"):
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        
        # Gemma tokenizer usually doesn't have a default pad token set in the json
        # We set it to <|pad|> or <|endoftext|> to avoid TypeError during batching
        if "<|pad|>" in self.tokenizer.get_vocab():
            self.tokenizer.pad_token = "<|pad|>"
        else:
            self.tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
            
        self.img_size = img_size
        self.mode = mode # 'multimodal' or 'text_only'
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def tokenize(self, text, max_len=512):
        # Prepend <bos> (id 2 for Gemma) to help model know where to start
        return self.tokenizer(
            "<bos>" + text + "<|endoftext|>",
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt"
        )["input_ids"].squeeze(0)

class HandwritingSource(JommarnMasterDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("Loading Handwriting Dataset...")
        self.data = load_dataset("iapp/thai_handwriting_dataset", split="train")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = self.transform(item["image"].convert("RGB"))
        
        # Wrap raw OCR text in the unified QA template
        prompt = "จงถอดความลายมือในรูปภาพนี้"
        text_format = f"ผู้ใช้: {prompt}\n\nผู้ช่วย: {item['text']}"
        tokens = self.tokenize(text_format)
        
        return img, tokens, tokens # img, input_ids, targets

class WikiSource(JommarnMasterDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("Loading Thai Wiki Dataset...")
        self.data = load_dataset("pythainlp/thai-wiki-dataset-v3", split="train")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Wiki has no image, we provide a dummy/zero tensor
        img = torch.zeros(3, self.img_size, self.img_size)
        tokens = self.tokenize(self.data[idx]["text"])
        return img, tokens, tokens

class AppenOCRSource(JommarnMasterDataset):
    def __init__(self, base_path="/kaggle/input/ocr-image-data-for-thai-documents", **kwargs):
        super().__init__(**kwargs)
        print(f"Checking Appen OCR at {base_path}...")
        # Note: Exact folder structure depends on Kaggle extraction
        self.img_files = glob.glob(os.path.join(base_path, "**/*.jpg"), recursive=True)
        # Mocking labels since access to specific JSON depends on user's Kaggle setup
        # In practice, you'd load the accompanying JSON/CSV for labels.

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = self.transform(Image.open(img_path).convert("RGB"))
        # Placeholder text: In training, replace with actual label from JSON
        tokens = self.tokenize("ตัวอย่างข้อความจากเอกสาร") 
        return img, tokens, tokens

class BlognoneNewsSource(JommarnMasterDataset):
    """
    Blognone Tech News Dataset (pythainlp/blognone_news)
    Text-only IT News.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("Loading Blognone News Dataset...")
        self.data = load_dataset("pythainlp/blognone_news", split="train")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = torch.zeros(3, self.img_size, self.img_size)
        
        task = random.choice(["ข่าวไอที", "รายงานข่าวเทคโนโลยี", "สรุปข่าว Blognone", "ข่าวล่าสุด"])
        prompt = task
        target = item["txt"]
        
        text_format = f"ผู้ใช้: {prompt}\n\nผู้ช่วย: {target}"
        tokens = self.tokenize(text_format)
        return img, tokens, tokens

class ThaiCulturaxSource(JommarnMasterDataset):
    """
    Thai CulturaX Clean Dataset (pythainlp/thai-culturax-clean-dataset)
    Massive Web Text Corpus (11.2M rows).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("Loading Thai CulturaX Dataset...")
        self.data = load_dataset("pythainlp/thai-culturax-clean-dataset", split="train")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = torch.zeros(3, self.img_size, self.img_size)
        
        task = random.choice(["อ่านบทความ", "ข้อความจากเว็บ", "เอกสารภาษาไทย", "ข้อมูลอินเทอร์เน็ต"])
        prompt = task
        target = item["text"]
        
        text_format = f"ผู้ใช้: {prompt}\n\nผู้ช่วย: {target}"
        tokens = self.tokenize(text_format)
        return img, tokens, tokens

class ThaiSumSource(JommarnMasterDataset):
    """
    Thai Text Summarization Dataset (AIAT/Pangpuriye-public_ThaiSum40k)
    Instruction Tuning for Summarization.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("Loading ThaiSum Dataset...")
        self.data = load_dataset("AIAT/Pangpuriye-public_ThaiSum40k", split="train")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = torch.zeros(3, self.img_size, self.img_size)
        
        prompt = f"{item['instruction']}\n\n{item['input']}"
        target = item["output"]
        
        text_format = f"ผู้ใช้: {prompt}\n\nผู้ช่วย: {target}"
        tokens = self.tokenize(text_format)
        return img, tokens, tokens

class DetailedOcrSource(JommarnMasterDataset):
    """
    Detailed OCR Source (Phonsiri/handwrite-ocr-detailed)
    Randomly select a task configuration to train on all relationships.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("Loading Phonsiri/handwrite-ocr-detailed Dataset...")
        self.data = load_dataset("Phonsiri/handwrite-ocr-detailed", split="train")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Optimization: Skip image loading if in text-only mode
        is_text_only = getattr(self, 'mode', 'multimodal') == 'text_only'
        if is_text_only:
            img = torch.zeros(3, self.img_size, self.img_size)
        else:
            try:
                img = self.transform(item["image"].convert("RGB"))
            except Exception:
                img = torch.zeros(3, self.img_size, self.img_size)
        
        # Select tasks based on mode
        if is_text_only:
            # User requested to exclude OCR column in text-only mode
            task = random.choice(["caption_th", "caption_en", "think_only"])
        else:
            task = random.choice(["ocr_only", "caption_th", "caption_en", "thinking_ocr"])
        
        if task == "ocr_only":
            prompt = "จงอ่านข้อความทั้งหมดที่ปรากฏอยู่ในภาพนี้"
            target = item["ocr_text"]
        elif task == "caption_th":
            prompt = "จงอธิบายรายละเอียดของภาพหรือเอกสารนี้อย่างละเอียด"
            target = item["detailed_caption_thai"]
        elif task == "caption_en":
            prompt = "Describe the details of this image or document in English."
            target = item["detailed_caption_en"]
        elif task == "think_only":
            prompt = "จงวิเคราะห์โครงสร้างของภาพนี้ทีละขั้นตอน"
            target = f"<think>\n{item.get('think_process', '')}\n</think>\n\nสรุป: {item.get('detailed_caption_thai', '')}"
        else: # thinking_ocr
            prompt = "จงวิเคราะห์ภาพและถอดข้อความภาษาไทยออกมาทีละขั้นตอน"
            target = (
                f"<think>\n{item.get('think_process', '')}\n</think>\n\n"
                f"**ข้อความที่ถอดได้:**\n{item.get('ocr_text', '')}\n\n"
                f"**คำบรรยายภาพ:**\n{item.get('detailed_caption_thai', '')}"
            )
            
        text_format = f"ผู้ใช้: {prompt}\n\nผู้ช่วย: {target}"
        tokens = self.tokenize(text_format)
        return img, tokens, tokens

class PdfOcrDetailedSource(JommarnMasterDataset):
    """
    Detailed OCR Source for PDFs (Phonsiri/pdf-ocr-detailed)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("Loading Phonsiri/pdf-ocr-detailed Dataset...")
        self.data = load_dataset("Phonsiri/pdf-ocr-detailed", split="train")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        is_text_only = getattr(self, 'mode', 'multimodal') == 'text_only'
        if is_text_only:
            img = torch.zeros(3, self.img_size, self.img_size)
        else:
            try:
                img = self.transform(item["image"].convert("RGB"))
            except Exception:
                img = torch.zeros(3, self.img_size, self.img_size)
        
        if is_text_only:
            task = random.choice(["caption_th", "caption_en", "think_only"])
        else:
            task = random.choice(["ocr_only", "caption_th", "caption_en", "thinking_ocr"])
        
        if task == "ocr_only":
            prompt = "จงอ่านข้อความทั้งหมดที่ปรากฏอยู่ในภาพหรือเอกสารนี้"
            target = item["ocr_text"]
        elif task == "caption_th":
            prompt = "จงอธิบายรายละเอียดของเอกสารนี้อย่างละเอียด"
            target = item["detailed_caption_thai"]
        elif task == "caption_en":
            prompt = "Describe the details of this document in English."
            target = item["detailed_caption_en"]
        elif task == "think_only":
            prompt = "จงวิเคราะห์โครงสร้างของเอกสารนี้ทีละขั้นตอน"
            target = f"<think>\n{item.get('think_process', '')}\n</think>\n\nสรุป: {item.get('detailed_caption_thai', '')}"
        else: # thinking_ocr
            prompt = "จงวิเคราะห์ภาพและถอดข้อความภาษาไทยออกมาทีละขั้นตอน"
            target = (
                f"<think>\n{item.get('think_process', '')}\n</think>\n\n"
                f"**ข้อความที่ถอดได้:**\n{item.get('ocr_text', '')}\n\n"
                f"**คำบรรยายภาพ:**\n{item.get('detailed_caption_thai', '')}"
            )
            
        text_format = f"ผู้ใช้: {prompt}\n\nผู้ช่วย: {target}"
        tokens = self.tokenize(text_format)
        return img, tokens, tokens


class AstrologyDatasetSource(JommarnMasterDataset):
    """
    Astrology & Document Layout Dataset (Phonsiri/astrology-dataset-clean)
    Provides OCR, Layout Captioning, and Category Classification.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("Loading Phonsiri/astrology-dataset-clean Dataset...")
        self.data = load_dataset("Phonsiri/astrology-dataset-clean", split="train")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = self.transform(item["image"].convert("RGB"))
        
        # Randomly select a task to train on different document dimensions
        task = random.choice(["ocr_text", "layout_caption", "classify"])
        
        if task == "ocr_text":
            prompt = "จงอ่านและถอดข้อความทั้งหมดจากเอกสารหรือรูปภาพนี้"
            target = item["text"]
        elif task == "layout_caption":
            prompt = "จงอธิบายโครงสร้างและรายละเอียดของรูปภาพหรือเอกสารนี้อย่างละเอียด"
            target = item["caption"]
        else: # classify
            prompt = "รูปภาพนี้จัดอยู่ในหมวดหมู่อะไร"
            target = f"หมวดหมู่: {item['category']}"
            
        text_format = f"ผู้ใช้: {prompt}\n\nผู้ช่วย: {target}"
        tokens = self.tokenize(text_format)
        return img, tokens, tokens

class CocoThaiDetailedSource(JommarnMasterDataset):
    """
    COCO Thai Detailed Captions Dataset (Phonsiri/coco-thai-gemma4-detailed)
    Provides General Scene Understanding in Thai and English with dynamic fallback cleaning.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("Loading Phonsiri/coco-thai-gemma4-detailed Dataset...")
        self.data = load_dataset("Phonsiri/coco-thai-gemma4-detailed", split="train")

    def __len__(self):
        return len(self.data)

    def _clean_text(self, text):
        if not text:
            return ""
        text_lower = text.lower()
        # Skip placeholders & instructions
        if "thai description" in text_lower or "english description" in text_lower:
            return ""
        if "combine all" in text_lower or "translate the" in text_lower:
            return ""
        return text.strip()

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Optimization: Skip image loading if in text-only mode
        if getattr(self, 'mode', 'multimodal') == 'text_only':
            img = torch.zeros(3, self.img_size, self.img_size)
        else:
            try:
                img = self.transform(item["image"].convert("RGB"))
            except Exception:
                img = torch.zeros(3, self.img_size, self.img_size)
        
        # Clean candidates
        th_detailed = self._clean_text(item.get("detailed_caption_thai"))
        en_detailed = self._clean_text(item.get("detailed_caption_en"))
        th_think = self._clean_text(item.get("think_process"))
        th_orig = self._clean_text(item.get("original_caption"))
        
        # Combine Thai and English detailed captions
        captions = []
        
        # Resolve best Thai caption
        th_caption = th_detailed if (th_detailed and len(th_detailed) > 10) else th_think
        if not th_caption or len(th_caption) < 10:
            th_caption = th_orig
            
        en_caption = en_detailed
        
        if th_caption:
            captions.append(th_caption)
            
        if en_caption and len(en_caption) > 10 and en_caption not in captions:
            # Check if one is a substring of the other to avoid duplicate/mixed sentences
            if th_caption and (th_caption in en_caption or en_caption in th_caption):
                # Use the longer one
                if len(en_caption) > len(th_caption):
                    captions = [en_caption]
                else:
                    captions = [th_caption]
            else:
                captions.append(en_caption)
                
        if not captions:
            # Absolute fallback
            captions = [th_orig if th_orig else "รายละเอียดรูปภาพ"]
            
        target = "\n\n".join(captions)
        prompt = "จงบรรยายรายละเอียดของภาพนี้อย่างละเอียด"
        
        text_format = f"ผู้ใช้: {prompt}\n\nผู้ช่วย: {target}"
        tokens = self.tokenize(text_format, max_len=512)
        return img, tokens, tokens

class DistillationSource(JommarnMasterDataset):
    """
    Teacher Distillation Dataset (Warmup Data)
    Reads from the pre-generated JSONL file containing Typhoon-OCR-7B responses.
    """
    def __init__(self, data_path="data/distilled_warmup_data.jsonl", **kwargs):
        super().__init__(**kwargs)
        print(f"Loading Distillation Dataset from {data_path}...")
        self.items = []
        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.items.append(json.loads(line))
        else:
            print(f"⚠️ Warning: Distillation file not found at {data_path}. Please run create_distillation_dataset.py first.")

    def __len__(self):
        return len(self.items) if self.items else 1 # Avoid division by zero

    def __getitem__(self, idx):
        if not self.items:
            # Fallback if file doesn't exist yet
            img = torch.zeros(3, self.img_size, self.img_size)
            tokens = self.tokenize("ผู้ใช้: จงวิเคราะห์ภาพนี้\n\nผู้ช่วย: ยังไม่มีข้อมูล Distillation")
            return img, tokens, tokens
            
        item = self.items[idx]
        try:
            img = self.transform(Image.open(item["image_path"]).convert("RGB"))
        except Exception:
            img = torch.zeros(3, self.img_size, self.img_size)
            
        prompt = "จงวิเคราะห์ภาพและถอดข้อความภาษาไทยออกมาทีละขั้นตอน"
        target = item["teacher_text"]
            
        text_format = f"ผู้ใช้: {prompt}\n\nผู้ช่วย: {target}"
        tokens = self.tokenize(text_format)
        return img, tokens, tokens

class ThaiOldBooksSource(JommarnMasterDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("Loading Thai Old Books Dataset...")
        self.data = load_dataset("pythainlp/thai-oldbooks", split="train")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Text-only dataset, so image is a dummy zero tensor
        img = torch.zeros(3, self.img_size, self.img_size)
        item = self.data[idx]
        
        # Add a nice format so the model learns literary style
        prompt = f"บทประพันธ์เรื่อง {item.get('book', 'ไม่ทราบชื่อเรื่อง')} โดย {item.get('author', 'ไม่ทราบนามปากกา')}"
        text_format = f"ผู้ใช้: {prompt}\n\nผู้ช่วย: {item.get('text', '')}"
        
        tokens = self.tokenize(text_format)
        return img, tokens, tokens

class JusciWebsiteSource(JommarnMasterDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("Loading Jusci Science News Dataset...")
        self.data = load_dataset("pythainlp/jusci-website", split="train")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Text-only dataset, so image is a dummy zero tensor
        img = torch.zeros(3, self.img_size, self.img_size)
        item = self.data[idx]
        
        # Add a nice format so the model learns news/article structures
        prompt = f"ข่าววิทยาศาสตร์เรื่อง: {item.get('title', 'ไม่ทราบชื่อเรื่อง')}"
        text_format = f"ผู้ใช้: {prompt}\n\nผู้ช่วย: {item.get('content', '')}"
        
        tokens = self.tokenize(text_format)
        return img, tokens, tokens

class WangchanLionWebSource(JommarnMasterDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("Loading WangchanLION-Web Dataset via STREAMING (to save disk space)...")
        # Use streaming=True to prevent 48GB disk download
        self.dataset_stream = load_dataset("aisingapore/WangchanLION-Web", split="train", streaming=True)
        self.iterable_data = iter(self.dataset_stream)
        # Fake length so WeightedRandomSampler can still assign probability weight
        self.fake_length = 5000000 

    def __len__(self):
        return self.fake_length

    def __getitem__(self, idx):
        # Text-only dataset, so image is a dummy zero tensor
        img = torch.zeros(3, self.img_size, self.img_size)
        
        # Ignore idx, just pull the next item from the stream
        try:
            item = next(self.iterable_data)
        except StopIteration:
            # If we somehow hit the end, restart the stream
            self.iterable_data = iter(self.dataset_stream)
            item = next(self.iterable_data)
        
        # General web text format
        prompt = "จงอ่านข้อความทั่วไปจากอินเทอร์เน็ต"
        text_format = f"ผู้ใช้: {prompt}\n\nผู้ช่วย: {item.get('text', '')}"
        
        tokens = self.tokenize(text_format)
        return img, tokens, tokens

class EnglishWikiSource(JommarnMasterDataset):
    """
    English WikiText Dataset (wikitext-103-v1)
    Provides basic English structure and grammar.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("Loading English WikiText Dataset...")
        self.data = load_dataset("wikitext", "wikitext-103-v1", split="train")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = torch.zeros(3, self.img_size, self.img_size)
        
        prompt = "Read the following English text:"
        target = item["text"].strip()
        if not target:
            target = "This section intentionally left blank."
            
        text_format = f"ผู้ใช้: {prompt}\n\nผู้ช่วย: {target}"
        tokens = self.tokenize(text_format)
        return img, tokens, tokens

def get_master_loader(batch_size=16, phase="multimodal"):
    """
    Creates a combined loader.
    If phase == 'text_only', it only loads Thai Wikipedia to pre-train the language model.
    If phase == 'multimodal', it balances across all vision-language sources.
    """
    if phase == "text_only":
        print("🚀 PHASE 1: TEXT ONLY (Language Pre-training)")
        ds_wiki = WikiSource(mode="text_only")
        ds_oldbooks = ThaiOldBooksSource(mode="text_only")
        ds_jusci = JusciWebsiteSource(mode="text_only")
        ds_blognone = BlognoneNewsSource(mode="text_only")
        ds_wangchan = WangchanLionWebSource(mode="text_only")
        ds_culturax = ThaiCulturaxSource(mode="text_only")
        ds_thaisum = ThaiSumSource(mode="text_only")
        ds_coco_text = CocoThaiDetailedSource(mode="text_only")
        ds_detailed_text = DetailedOcrSource(mode="text_only")
        ds_pdf_detailed_text = PdfOcrDetailedSource(mode="text_only")
        ds_enwiki = EnglishWikiSource(mode="text_only")
        
        master_ds_text = ConcatDataset([
            ds_wiki, ds_oldbooks, ds_jusci, ds_blognone, ds_wangchan, ds_culturax, ds_thaisum,
            ds_coco_text, ds_detailed_text, ds_pdf_detailed_text, ds_enwiki
        ])
        
        num_wiki = len(ds_wiki)
        num_oldbooks = len(ds_oldbooks)
        num_jusci = len(ds_jusci)
        num_blognone = len(ds_blognone)
        num_wangchan = len(ds_wangchan)
        num_culturax = len(ds_culturax)
        num_thaisum = len(ds_thaisum)
        num_coco = len(ds_coco_text)
        num_detailed = len(ds_detailed_text)
        num_pdf_detailed = len(ds_pdf_detailed_text)
        num_enwiki = len(ds_enwiki)
        total_text = num_wiki + num_oldbooks + num_jusci + num_blognone + num_wangchan + num_culturax + num_thaisum + num_coco + num_detailed + num_pdf_detailed + num_enwiki
        
        # Weight distribution
        dataset_lengths = [
            num_wiki, num_oldbooks, num_jusci, num_blognone, num_wangchan, 
            num_culturax, num_thaisum, num_coco, num_detailed, num_pdf_detailed, num_enwiki
        ]
        dataset_weights = [
            0.15, 0.05, 0.05, 0.05, 0.25,
            0.25, 0.05, 0.05, 0.02, 0.03, 0.05
        ]
        
        sampler = HugeWeightedRandomSampler(dataset_lengths, dataset_weights, num_samples=total_text)
        
        print(f"Text-Only Balanced Sampler Active:")
        print(f" - WangchanLION-Web: {num_wangchan} rows (Sample weight: 25%)")
        print(f" - Thai CulturaX: {num_culturax} rows (Sample weight: 25%)")
        print(f" - Thai Wiki: {num_wiki} rows (Sample weight: 15%)")
        print(f" - English WikiText: {num_enwiki} rows (Sample weight: 5%)")
        print(f" - ThaiSum (Summarization): {num_thaisum} rows (Sample weight: 5%)")
        print(f" - Jusci Science News: {num_jusci} rows (Sample weight: 5%)")
        print(f" - Blognone Tech News: {num_blognone} rows (Sample weight: 5%)")
        print(f" - COCO Descriptions: {num_coco} rows (Sample weight: 5%)")
        print(f" - Thai Old Books: {num_oldbooks} rows (Sample weight: 5%)")
        print(f" - PDF Logic: {num_pdf_detailed} rows (Sample weight: 3%)")
        print(f" - Handwriting Logic: {num_detailed} rows (Sample weight: 2%)")
        
        return DataLoader(
            master_ds_text, 
            batch_size=batch_size, 
            sampler=sampler,
            num_workers=2,
            pin_memory=True
        )
        
    # Phase 2: Multimodal Balanced Load
    print("🌌 PHASE 2: MULTIMODAL ALIGNMENT")
    ds_hw = HandwritingSource()
    ds_wiki = WikiSource()
    ds_oldbooks = ThaiOldBooksSource()
    ds_jusci = JusciWebsiteSource()
    ds_blognone = BlognoneNewsSource()
    ds_wangchan = WangchanLionWebSource()
    ds_culturax = ThaiCulturaxSource()
    ds_thaisum = ThaiSumSource()
    ds_detailed = DetailedOcrSource()
    ds_pdf_detailed = PdfOcrDetailedSource()
    ds_astrology = AstrologyDatasetSource()
    ds_coco = CocoThaiDetailedSource()
    ds_distill = DistillationSource()
    ds_enwiki = EnglishWikiSource()
    
    # Combined dataset
    master_ds = ConcatDataset([
        ds_hw, ds_wiki, ds_oldbooks, ds_jusci, ds_blognone, ds_wangchan, ds_culturax, ds_thaisum,
        ds_detailed, ds_pdf_detailed, ds_astrology, ds_coco, ds_distill, ds_enwiki
    ])
    
    num_hw = len(ds_hw)
    num_wiki = len(ds_wiki)
    num_oldbooks = len(ds_oldbooks)
    num_jusci = len(ds_jusci)
    num_blognone = len(ds_blognone)
    num_wangchan = len(ds_wangchan)
    num_culturax = len(ds_culturax)
    num_thaisum = len(ds_thaisum)
    num_detailed = len(ds_detailed)
    num_pdf_detailed = len(ds_pdf_detailed)
    num_astrology = len(ds_astrology)
    num_coco = len(ds_coco)
    num_distill = len(ds_distill)
    num_enwiki = len(ds_enwiki)
    total = num_hw + num_wiki + num_oldbooks + num_jusci + num_blognone + num_wangchan + num_culturax + num_thaisum + num_detailed + num_pdf_detailed + num_astrology + num_coco + num_distill + num_enwiki
    
    # Weight Distribution (Distillation takes 40% of the batches to stabilize learning fast)
    # Dynamic check: If distillation data is not ready, redistribute its weight
    has_distill = len(getattr(ds_distill, 'items', [])) > 0
    
    if has_distill:
        base_vision_weight = 0.10
        distill_weight = 0.40
    else:
        print("⚠️ No Distillation data found. Redistributing its 40% weight to other vision datasets.")
        base_vision_weight = 0.18 # 0.10 + (0.40 / 5)
        distill_weight = 0.0
        
    dataset_lengths = [
        num_hw, num_wiki, num_oldbooks, num_jusci, num_blognone,
        num_wangchan, num_culturax, num_thaisum, num_detailed,
        num_pdf_detailed, num_astrology, num_coco, num_distill, num_enwiki
    ]
    dataset_weights = [
        base_vision_weight, 0.01, 0.01, 0.01, 0.01,
        0.02, 0.02, 0.01, base_vision_weight,
        base_vision_weight, base_vision_weight, base_vision_weight, distill_weight, 0.01
    ]
    
    sampler = HugeWeightedRandomSampler(dataset_lengths, dataset_weights, num_samples=total)
    
    print(f"Balanced Sampler Active:")
    print(f" - Distillation (Teacher Data): {num_distill} rows (Sample weight: {distill_weight*100:.0f}%)")
    print(f" - Detailed OCR: {num_detailed} rows (Sample weight: {base_vision_weight*100:.0f}%)")
    print(f" - PDF OCR: {num_pdf_detailed} rows (Sample weight: {base_vision_weight*100:.0f}%)")
    print(f" - Astrology Layout: {num_astrology} rows (Sample weight: {base_vision_weight*100:.0f}%)")
    print(f" - COCO General: {num_coco} rows (Sample weight: {base_vision_weight*100:.0f}%)")
    print(f" - Handwriting: {num_hw} rows (Sample weight: {base_vision_weight*100:.0f}%)")
    print(f" - Thai Wiki: {num_wiki} rows (Sample weight: 1%)")
    print(f" - English WikiText: {num_enwiki} rows (Sample weight: 1%)")
    print(f" - WangchanLION-Web: {num_wangchan} rows (Sample weight: 2%)")
    print(f" - Thai CulturaX: {num_culturax} rows (Sample weight: 2%)")
    print(f" - ThaiSum (Summarization): {num_thaisum} rows (Sample weight: 1%)")
    print(f" - Blognone Tech News: {num_blognone} rows (Sample weight: 1%)")
    print(f" - Jusci Science News: {num_jusci} rows (Sample weight: 1%)")
    print(f" - Thai Old Books: {num_oldbooks} rows (Sample weight: 1%)")

    return DataLoader(
        master_ds, 
        batch_size=batch_size, 
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )

if __name__ == "__main__":
    # Test
    try:
        loader = get_master_loader(batch_size=4)
        img, input_ids, targets = next(iter(loader))
        print(f"Batch Loaded Success!")
        print(f"Images: {img.shape}, Tokens: {input_ids.shape}")
    except Exception as e:
        print(f"Error: {e}")
        print("Note: This script is designed to run in an environment with access to the datasets.")