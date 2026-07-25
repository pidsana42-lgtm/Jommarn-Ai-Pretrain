import os
import json
import torch
import base64
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

def setup_teacher():
    print("Loading Typhoon-OCR-7B Teacher Model...")
    model_id = "scb10x/typhoon-ocr-7b"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model in bfloat16 to save memory (requires ~14GB VRAM)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).eval()
    
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor, device

def pil_to_base64(img):
    buffered = BytesIO()
    # Convert RGBA to RGB if necessary
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def get_typhoon_prompt():
    # Using the 'default' prompt from Typhoon OCR specs
    return (
        "Below is an image of a document page along with its dimensions. "
        "Simply return the markdown representation of this document, presenting tables in markdown format as they naturally appear.\n"
        "If the document contains images, use a placeholder like dummy.png for each image.\n"
        "Your final output must be in JSON format with a single key `natural_text` containing the response.\n"
        "RAW_TEXT_START\n\nRAW_TEXT_END"
    )

def generate_teacher_response(model, processor, device, image_pil):
    prompt = get_typhoon_prompt()
    img_b64 = pil_to_base64(image_pil)
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        ],
    }]
    
    # Apply chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(
        text=[text],
        images=[image_pil],
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            temperature=0.1,
            max_new_tokens=2048,
            num_return_sequences=1,
            repetition_penalty=1.2,
            do_sample=True,
        )
        
    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = output[:, prompt_length:]
    text_output = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
    
    # Extract the natural_text from the JSON output
    try:
        # Sometimes the model outputs markdown code blocks like ```json ... ```
        clean_text = text_output.replace("```json", "").replace("```", "").strip()
        parsed_json = json.loads(clean_text)
        return parsed_json.get("natural_text", text_output)
    except json.JSONDecodeError:
        return text_output

import argparse

def create_distillation_dataset(is_test=False):
    model, processor, device = setup_teacher()
    
    print("Loading datasets to distill...")
    # If testing, just do 2 samples. Otherwise 500 each.
    samples_per_ds = 2 if is_test else 500
    datasets_to_process = [
        ("Phonsiri/handwrite-ocr-detailed", samples_per_ds),
        ("Phonsiri/astrology-dataset-clean", samples_per_ds)
    ]
    
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    
    filename = "distilled_warmup_data_test.jsonl" if is_test else "distilled_warmup_data.jsonl"
    out_path = os.path.join(out_dir, filename)
    
    print(f"Generating distilled data to {out_path}...")
    
    with open(out_path, "w", encoding="utf-8") as f:
        item_id = 0
        for ds_name, num_samples in datasets_to_process:
            print(f"Processing {ds_name}...")
            ds = load_dataset(ds_name, split="train")
            
            # Shuffle so we get a random subset
            ds = ds.shuffle(seed=42)
            
            for i in tqdm(range(min(num_samples, len(ds)))):
                item = ds[i]
                img = item["image"]
                
                try:
                    # Extract raw markdown from Typhoon
                    raw_markdown = generate_teacher_response(model, processor, device, img)
                    
                    # 🔥 MAGIC HERE: Format it like a normal chatbot conversation
                    chatbot_style_response = f"จากภาพที่เห็น ฉันสามารถถอดข้อความออกมาได้ดังนี้ครับ:\n\n{raw_markdown}\n\nหากต้องการให้ฉันวิเคราะห์ส่วนไหนเพิ่มเติม บอกได้เลยนะครับ!"
                    
                    # Save the image locally to keep it paired with the text
                    img_filename = f"distill_{item_id}.jpg"
                    img_save_path = os.path.join(out_dir, img_filename)
                    if img.mode in ('RGBA', 'P'):
                        img = img.convert('RGB')
                    img.save(img_save_path)
                    
                    record = {
                        "id": item_id,
                        "source": ds_name,
                        "image_path": img_save_path,
                        "teacher_text": chatbot_style_response
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    item_id += 1
                except Exception as e:
                    print(f"Error processing item {i} in {ds_name}: {e}")
                    
    print(f"Successfully generated {item_id} distilled training pairs!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run a quick test with 2 samples")
    args = parser.parse_args()
    create_distillation_dataset(is_test=args.test)
