import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from config.config import default_config as config
from src.models.transformer import JommarnOmni as Transformer
from scripts.master_data_loader import get_master_loader
from typing import Dict

# --- Initialize the Model and Print Parameters ---

model = Transformer(
    n_head=config['n_head'],
    n_embed=config['n_embed'],
    context_length=config['context_length'],
    vocab_size=config['vocab_size'],
    N_BLOCKS=config['n_blocks'],
    n_kv_head=config['n_kv_heads'],
    v_layers=config.get('v_layers', 12)
)

# Multi-GPU Support
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for maximum acceleration!")
    model = torch.nn.DataParallel(model)

model = model.to(config['device'])

# --- Resume Training Logic ---
start_step = 0
hf_repo = os.getenv("HF_REPO_ID", "Jommarn/jommarn-omni-checkpoints")
force_reset = os.getenv("FORCE_RESET") == "1"
checkpoint_name = os.path.basename(config['t_out_path']).replace(".pt", "_latest.pt")
local_checkpoint_path = os.path.join("models", checkpoint_name)

if force_reset:
    print("FORCE_RESET=1: Starting training from scratch (Step 0).")
else:
    # 1. Try to download from Hugging Face if not found locally
    if hf_repo and not os.path.exists(local_checkpoint_path):
        print(f"Checking for latest checkpoint in Hugging Face Hub: {hf_repo}...")
        try:
            from huggingface_hub import hf_hub_download
            os.makedirs("models", exist_ok=True)
            downloaded_path = hf_hub_download(
                repo_id=hf_repo, 
                filename=checkpoint_name,
                local_dir="models",
                local_dir_use_symlinks=False
            )
            print(f"Downloaded checkpoint from Hub: {downloaded_path}")
        except Exception as e:
            print(f"No checkpoint found on Hub or error: {e}")

    # 2. Load the latest checkpoint
    if os.path.exists(local_checkpoint_path):
        print(f"Resuming training from checkpoint: {local_checkpoint_path}")
        try:
            # ✅ เพิ่ม weights_only=False เพื่อให้โหลด Optimizer/Scheduler ได้ใน PyTorch 2.6+
            checkpoint = torch.load(local_checkpoint_path, map_location=config['device'], weights_only=False)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            
            # Load state dict with handling for module prefix
            if hasattr(model, 'module'):
                model.module.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
            else:
                model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
                
            start_step = checkpoint.get('steps', 0)
            print(f"Restarting from step: {start_step}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Starting from scratch.")
            start_step = 0

# --- Optimizer and Stability Tools ---

optimizer = torch.optim.AdamW(model.parameters(), lr=config['t_lr'])
scaler = torch.amp.GradScaler('cuda')

# Learning Rate Scheduler with Warmup
def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scheduler = get_lr_scheduler(optimizer, warmup_steps=2000, total_steps=config['t_train_steps'])  # ยืด Warmup 2x เพื่อความเสถียรและกัน Loss Spike

if not force_reset and os.path.exists(local_checkpoint_path):
    try:
        # ✅ เพิ่ม weights_only=False เช่นกัน
        checkpoint = torch.load(local_checkpoint_path, map_location=config['device'], weights_only=False)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            # ✅ โหลด Scheduler State กลับมาด้วย (แก้บั๊ก LR Reset)
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            # ถ้า Checkpoint เก่าไม่มี scheduler state ให้เดิน scheduler ไปยัง step ปัจจุบัน
            print(f"No scheduler state found, fast-forwarding scheduler to step {start_step}...")
            for _ in range(start_step):
                scheduler.step()
    except Exception as e:
        print(f"Warning: Could not load optimizer/scheduler state: {e}")

losses = []
AVG_WINDOW = 64

# --- Multimodal Training Loop ---

train_phase = os.getenv("TRAIN_PHASE", "multimodal")
train_loader = get_master_loader(batch_size=config['t_batch_size'], phase=train_phase)
os.makedirs(os.path.dirname(config['t_out_path']), exist_ok=True)

print(f"Starting Training on {config['device']} with physical batch size {config['t_batch_size']}...")
grad_accum_steps = config.get('t_grad_accum', 1)
print(f"Gradient Accumulation: {grad_accum_steps} steps (Effective batch size: {config['t_batch_size'] * grad_accum_steps})")

pbar = tqdm(range(start_step, config['t_train_steps']))
train_iter = iter(train_loader)
inner_model = model.module if hasattr(model, 'module') else model
local_step = 0  # นับ Steps จากการ Resume เพื่อกันอัปโหลดทันที

for step in pbar:
    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0
    
    try:
        for _ in range(grad_accum_steps):
            try:
                images, tokens, _ = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                images, tokens, _ = next(train_iter)

            # Shift tokens for next-token prediction
            xb = tokens[:, :-1].to(config['device'])
            yb = tokens[:, 1:].to(config['device'])
            
            # Skip Vision Encoder completely if in Phase 1 (Text-Only)
            if train_phase == "text_only":
                images = None
            else:
                images = images.to(config['device'])

            with torch.amp.autocast('cuda'):
                logits, loss = model(xb, images=images, targets=yb)
            
            loss = loss / grad_accum_steps
            scaler.scale(loss).backward()
            accum_loss += loss.item()

        # Gradient Clipping & Optimizer Step
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # เข์มแน่นขึ้น 0.5 เพื่อกัน Gradient Explosion
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()  # ✅ อัปเดต LR ทุก Step (Warmup → Cosine Decay)
        
        losses.append(accum_loss)
        pbar.set_description(f"Loss: {np.mean(losses[-AVG_WINDOW:]):.4f}")
        local_step += 1

        # Checkpoints & Hub Sync
        if step > 0 and step % config['t_eval_steps'] == 0:
            step_path = config['t_out_path'].replace(".pt", f"_step_{step}.pt")
            torch.save(inner_model.state_dict(), step_path)
            
            # Cleanup: Keep only the last 2 step checkpoints to save disk space
            import glob
            checkpoint_dir = os.path.dirname(config['t_out_path']) or "."
            base_name = os.path.basename(config['t_out_path']).replace(".pt", "")
            saved_steps = sorted(glob.glob(os.path.join(checkpoint_dir, f"{base_name}_step_*.pt")), key=os.path.getmtime)
            
            while len(saved_steps) > 2:
                oldest = saved_steps.pop(0)
                try:
                    os.remove(oldest)
                except OSError:
                    pass
            
        # --- Checkpoint Save (ทุก 50 steps ลงดิสก์เครื่อง Cloud) ---
        if local_step > 0 and local_step % 50 == 0:
            temp_checkpoint = config['t_out_path'].replace(".pt", "_latest.pt")
            torch.save({
                'model_state_dict': inner_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),  # ✅ บันทึก Scheduler State ด้วย
                'steps': step,
                'losses': losses
            }, temp_checkpoint)

        # --- HuggingFace Sync (ทุก 100 steps) ---
        if local_step > 0 and local_step % 100 == 0 and hf_repo:
            try:
                from scripts.push_to_hf import push_to_hub
                push_to_hub(repo_id=hf_repo, model_path=temp_checkpoint)
            except Exception as e:
                print(f"HuggingFace Sync Failed: {e}")

    except Exception as e:
        print(f"Error at step {step}: {e}")
        continue

# Save Final
torch.save({'model_state_dict': inner_model.state_dict(), 'steps': config['t_train_steps']}, config['t_out_path'])
print(f"Training Complete. Saved to {config['t_out_path']}")