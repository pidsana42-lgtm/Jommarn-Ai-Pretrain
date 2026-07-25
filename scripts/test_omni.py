import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import PreTrainedTokenizerFast
from src.models.transformer import JommarnOmni as Transformer
import argparse
import os
from config.config import default_config as config

def test_omni(model_path, image_path, prompt, vocab_size=config['vocab_size'], n_embed=config['n_embed'], n_head=config['n_head'], n_blocks=config['n_blocks'], n_kv_head=config['n_kv_heads'], context_length=config['context_length'], max_new_tokens=300, temperature=0.8, top_p=0.9, repetition_penalty=1.2, img_size=512, v_layers=config.get('v_layers', 12)):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load Tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
    
    # 2. Load Model
    model = Transformer(
        n_head=n_head, 
        n_embed=n_embed, 
        context_length=context_length, 
        vocab_size=vocab_size, 
        N_BLOCKS=n_blocks,
        n_kv_head=n_kv_head,
        img_size=img_size,
        v_layers=v_layers
    )
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    # Handle DataParallel or raw state_dict
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # Clean state_dict if it was saved with DataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    
    # FREE MEMORY: Delete checkpoint containing heavy optimizer states
    del checkpoint
    del state_dict
    del new_state_dict
    torch.cuda.empty_cache()
    
    # CAST TO BFLOAT16 to cut VRAM usage in half!
    if device == 'cuda':
        model = model.bfloat16()
        
    model.to(device)
    model.eval()

    # 3. Process Image (Optional)
    image_tensor = None
    if image_path and os.path.exists(image_path):
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device) # (1, 3, img_size, img_size)
        print(f"Vision Mode: Image '{image_path}' attached (resized to {img_size}x{img_size}).")
    else:
        print("Text-Only Mode: No image attached.")

    # 4. Process Text
    # If the user didn't provide a chat template, wrap it automatically using unicode escapes to avoid encoding issues
    if "\u0e1c\u0e39\u0e49\u0e43\u0e0a\u0e49:" not in prompt and "\u0e1c\u0e39\u0e49\u0e0a\u0e48\u0e27\u0e22:" not in prompt:
        prompt = f"\u0e1c\u0e39\u0e49\u0e43\u0e0a\u0e49: {prompt}\n\n\u0e1c\u0e39\u0e49\u0e0a\u0e48\u0e27\u0e22: "
        print(f"Auto-formatted prompt: {prompt.strip()}")
        
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

    # 5. Generate with Advanced Sampling (Temperature, Top-k, Top-p)
    print(f"Jommarn-Omni is thinking...")
    max_new_tokens = max_new_tokens  # from CLI arg
    temperature = temperature        # from CLI arg
    top_k = 50                       # Filter low-probability tokens
    top_p = top_p                    # Nucleus sampling for diversity
    # Use context_length from function arguments
    max_text_ctx = context_length    # max text tokens in one step

    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.encode("<|endoftext|>")[0]

    # ✅ Fix 1: Pre-compute vision tokens ONCE outside loop (10-20x faster)
    with torch.no_grad():
        v_tokens = model.vision_encoder(image_tensor) if image_tensor is not None else None

    with torch.no_grad():
        generated_ids = input_ids
        
        # MTP generates 4 tokens per forward pass -> cut loop steps to a quarter!
        steps = max(1, max_new_tokens // 4)
        print(f"  Running {steps} MTP steps (Generating up to {steps * 4} tokens)...")
        
        for step in range(steps):
            # ✅ Fix 2: Truncate text context to stay within model's context window
            idx_cond = generated_ids[:, -max_text_ctx:]

            # Pass pre-computed v_tokens (avoids re-running vision encoder every step)
            logits, _ = model(idx_cond, v_tokens=v_tokens)
            
            # --- 1. PREDICT TOKEN 1 (Standard Head) ---
            logits_1 = logits[:, -1, :] / temperature

            if repetition_penalty != 1.0:
                for t_id in set(generated_ids[0].tolist()):
                    logits_1[0, t_id] = logits_1[0, t_id] / repetition_penalty if logits_1[0, t_id] > 0 else logits_1[0, t_id] * repetition_penalty

            # Top-k sampling for Token 1
            if top_k > 0:
                top_k_vals = torch.topk(logits_1, min(top_k, logits_1.size(-1)))[0][..., -1, None]
                logits_1[logits_1 < top_k_vals] = float('-inf')

            # Top-p (Nucleus) sampling for Token 1
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits_1, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits_1[:, indices_to_remove] = float('-inf')

            probs_1 = F.softmax(logits_1, dim=-1)
            next_id_1 = torch.multinomial(probs_1, num_samples=1)
            
            # Store generated tokens for this step
            step_gen_ids = [next_id_1]
            h_state = model._last_h.unsqueeze(1) # (B, 1, n_embed)
            
            # --- 2. PREDICT TOKENS 2, 3, AND 4 (MTP Recursive Heads) ---
            for k in range(3):
                # Embed the previous generated token
                emb_prev = model.token_embed(step_gen_ids[-1])
                # Mix using current MTP mixer
                h_state = model.mtp_mixers[k](torch.cat([h_state, emb_prev], dim=-1))
                # Project to vocab
                logits_k = model.lm_head(h_state).squeeze(1) / temperature
                
                if repetition_penalty != 1.0:
                    current_tokens = generated_ids[0].tolist() + [t[0].item() for t in step_gen_ids]
                    for t_id in set(current_tokens):
                        logits_k[0, t_id] = logits_k[0, t_id] / repetition_penalty if logits_k[0, t_id] > 0 else logits_k[0, t_id] * repetition_penalty
                
                # Apply Top-k to Token k+2
                if top_k > 0:
                    top_k_vals_k = torch.topk(logits_k, min(top_k, logits_k.size(-1)))[0][..., -1, None]
                    logits_k[logits_k < top_k_vals_k] = float('-inf')
                    
                # Apply Top-p to Token k+2
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits_k, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits_k[:, indices_to_remove] = float('-inf')
                    
                probs_k = F.softmax(logits_k, dim=-1)
                next_id_k = torch.multinomial(probs_k, num_samples=1)
                step_gen_ids.append(next_id_k)

            # Concatenate all 4 newly predicted tokens to sequence
            new_step_tokens = torch.cat(step_gen_ids, dim=1)
            generated_ids = torch.cat((generated_ids, new_step_tokens), dim=1)

            # Early stopping check if any generated token in this step is EOS (id 1) or has "<|endoftext|>"
            eos_detected = False
            for idx_in_step, tid in enumerate(step_gen_ids):
                token_text = tokenizer.decode(tid.item())
                if tid.item() == eos_token_id or "<|endoftext|>" in token_text:
                    print(f"  [EOS detected at step {step*4 + idx_in_step + 1}]")
                    eos_detected = True
                    break
            
            if eos_detected:
                break
        
    result = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
    
    # Post-process cleanup
    result = result.replace("<|endoftext|>", "").strip()
    # Remove any trailing junk or repeated ' คือ'
    print(f"\n--- Result ---\n{result}\n--------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/jommarn_omni_206m_l40s_latest.pt")
    parser.add_argument("--image", type=str, default=None, help="Path to test image (optional)")
    parser.add_argument("--prompt", type=str, default="รูปภาพนี้คือ", help="Thai prompt")
    parser.add_argument("--max_new_tokens", type=int, default=300, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (lower = more focused)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P (Nucleus) sampling (1.0 = off, 0.9 = confident words only)")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Penalty for repeating tokens (1.0 = off, >1.0 = penalize)")
    parser.add_argument("--n_kv_head", type=int, default=2, help="Number of KV heads (GQA)")
    parser.add_argument("--img_size", type=int, default=512, help="Image size (default 512)")

    args = parser.parse_args()
    test_omni(
        args.model, 
        args.image, 
        args.prompt, 
        max_new_tokens=args.max_new_tokens, 
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        n_kv_head=args.n_kv_head,
        img_size=args.img_size
    )