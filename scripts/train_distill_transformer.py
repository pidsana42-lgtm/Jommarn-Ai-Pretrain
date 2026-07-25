import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from config.config import default_config as config
from src.models.transformer import JommarnOmni as Transformer
from scripts.distill_data_loader import get_distill_loader

def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def main():
    print("=" * 60)
    print("🚀 JOMMARN-OMNI PRETRAINING WITH LOGITS DISTILLATION & 4-TOKEN MTP")
    print("=" * 60)
    
    # 1. Initialize Model
    model = Transformer(
        n_head=config['n_head'],
        n_embed=config['n_embed'],
        context_length=config['context_length'],
        vocab_size=config['vocab_size'],
        N_BLOCKS=config['n_blocks'],
        n_kv_head=config['n_kv_heads'],
        v_layers=config.get('v_layers', 12)
    )

    if torch.cuda.device_count() > 1:
        print(f"🔥 Using {torch.cuda.device_count()} GPUs for parallel distillation!")
        model = torch.nn.DataParallel(model)

    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 2. Checkpoint Resume & HuggingFace Sync Logic
    start_step = 0
    src_hf_repo = os.getenv("SRC_HF_REPO", "Phonsiri/jommarn-omni-checkpoints")
    push_hf_repo = os.getenv("HF_REPO_ID", "Jommarn/jommarn-omni-checkpoints")
    force_reset = os.getenv("FORCE_RESET") == "1"
    checkpoint_name = os.path.basename(config['t_out_path']).replace(".pt", "_latest.pt")
    local_checkpoint_path = os.path.join("models", checkpoint_name)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['t_lr'])
    scheduler = get_lr_scheduler(optimizer, warmup_steps=2000, total_steps=config['t_train_steps'])
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    if force_reset:
        print("⚡ FORCE_RESET=1: Starting training from scratch (Step 0).")
    else:
        # 2a. Download initial checkpoint from source Hugging Face Hub (Phonsiri) if missing locally
        if not os.path.exists(local_checkpoint_path):
            # First try push repo (Jommarn), if not found then fallback to source repo (Phonsiri)
            for target_repo in [push_hf_repo, src_hf_repo]:
                if target_repo:
                    print(f"📥 Checking and downloading checkpoint from Hugging Face Hub: {target_repo}...")
                    try:
                        from huggingface_hub import hf_hub_download
                        os.makedirs("models", exist_ok=True)
                        downloaded_path = hf_hub_download(
                            repo_id=target_repo,
                            filename=checkpoint_name,
                            local_dir="models",
                            local_dir_use_symlinks=False
                        )
                        print(f"✅ Successfully downloaded base checkpoint from {target_repo}: {downloaded_path}")
                        break
                    except Exception as e:
                        print(f"⚠️ Checkpoint not found in {target_repo} or error: {e}")

        # 2b. Load existing local checkpoint
        if os.path.exists(local_checkpoint_path):
            print(f"🔄 Resuming training from checkpoint: {local_checkpoint_path}")
            try:
                checkpoint = torch.load(local_checkpoint_path, map_location=device, weights_only=False)
                raw_state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
                
                inner_model = model.module if hasattr(model, 'module') else model
                current_state = inner_model.state_dict()
                
                # Filter out buffer size mismatches (e.g. rope_cos, rope_sin, tril due to context length change)
                clean_state_dict = {}
                for k, v in raw_state.items():
                    key = k.replace('module.', '')
                    if key in current_state:
                        if current_state[key].shape == v.shape:
                            clean_state_dict[key] = v
                        else:
                            print(f"⏩ Ignoring buffer size mismatch for {key}: checkpoint {v.shape} vs model {current_state[key].shape}")
                    else:
                        clean_state_dict[key] = v
                
                missing_keys, unexpected_keys = inner_model.load_state_dict(clean_state_dict, strict=False)
                print(f"🎉 Successfully loaded model weights! (Loaded {len(clean_state_dict)} matching layers)")
                
                if 'optimizer_state_dict' in checkpoint:
                    try:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        print("✅ Loaded optimizer state!")
                    except Exception as opt_err:
                        print(f"⚠️ Optimizer state shape changed due to context length tuning: starting fresh optimizer state.")
                if 'scheduler_state_dict' in checkpoint:
                    try:
                        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    except Exception:
                        pass
                else:
                    for _ in range(checkpoint.get('steps', 0)):
                        scheduler.step()
                        
                start_step = checkpoint.get('steps', 0)
                print(f"🎉 Successfully resumed training at Step: {start_step}!")
            except Exception as e:
                print(f"⚠️ Failed to load checkpoint: {e}. Starting from scratch.")
                start_step = 0

    # 3. Hyperparameters & Distillation Config
    dataset_name = os.getenv("DISTILL_DATASET", "Jommarn/qwen2.5-7b-pretrain-distilled-logits")
    alpha_distill = float(os.getenv("ALPHA_DISTILL", "0.5"))   # 50% Hard Loss + 50% Soft Logits Loss
    distill_temp = float(os.getenv("DISTILL_TEMP", "1.0"))     # Softmax temperature for distillation
    
    print(f"📊 Distillation Dataset: {dataset_name}")
    print(f"⚙️ Alpha Distill (Soft Loss Ratio): {alpha_distill}")
    print(f"🌡️ Distillation Temperature: {distill_temp}")

    # 4. Load Data Loader
    train_loader = get_distill_loader(
        dataset_name=dataset_name,
        batch_size=config['t_batch_size'],
        context_length=config['context_length']
    )
    
    grad_accum_steps = config.get('t_grad_accum', 1)
    train_iter = iter(train_loader)
    losses = []
    local_step = 0
    inner_model = model.module if hasattr(model, 'module') else model
    
    distill_steps = int(os.getenv("STEPS", "2000"))
    target_total_steps = start_step + distill_steps
    
    print(f"⚡ Starting Logits Distillation Pretraining from Step {start_step} to {target_total_steps} (+{distill_steps} steps) on {device}...")
    pbar = tqdm(range(start_step, target_total_steps))
    
    for step in pbar:
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        
        for _ in range(grad_accum_steps):
            try:
                xb, yb, t_logits, t_ids = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                xb, yb, t_logits, t_ids = next(train_iter)
                
            xb = xb.to(device)
            yb = yb.to(device)
            t_logits = t_logits.to(device)
            t_ids = t_ids.to(device)
            
            if torch.cuda.is_available() and scaler is not None:
                with torch.amp.autocast('cuda'):
                    logits, loss = model(
                        xb, 
                        targets=yb, 
                        teacher_top_logits=t_logits, 
                        teacher_top_ids=t_ids,
                        alpha_distill=alpha_distill,
                        distill_temp=distill_temp
                    )
                loss = loss / grad_accum_steps
                scaler.scale(loss).backward()
            else:
                logits, loss = model(
                    xb, 
                    targets=yb, 
                    teacher_top_logits=t_logits, 
                    teacher_top_ids=t_ids,
                    alpha_distill=alpha_distill,
                    distill_temp=distill_temp
                )
                loss = loss / grad_accum_steps
                loss.backward()
                
            accum_loss += loss.item()
            
        if torch.cuda.is_available() and scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
        scheduler.step()
        losses.append(accum_loss)
        local_step += 1
        pbar.set_description(f"Distill Loss: {np.mean(losses[-32:]):.4f}")

        # Checkpoint Saving & HF Uploading (Every 50 / 100 steps)
        if local_step > 0 and local_step % 50 == 0:
            os.makedirs("models", exist_ok=True)
            temp_checkpoint = config['t_out_path'].replace(".pt", "_latest.pt")
            torch.save({
                'model_state_dict': inner_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'steps': step,
                'losses': losses
            }, temp_checkpoint)
            
        if local_step > 0 and local_step % 100 == 0 and push_hf_repo:
            try:
                from scripts.push_to_hf import push_to_hub
                temp_checkpoint = config['t_out_path'].replace(".pt", "_latest.pt")
                push_to_hub(repo_id=push_hf_repo, model_path=temp_checkpoint)
                print(f"☁️ Uploaded latest checkpoint at step {step} to HF Hub ({push_hf_repo})!")
            except Exception as e:
                print(f"⚠️ HF Sync Failed: {e}")
        
    print("🎉 Logits Distillation Pretraining Run Complete!")

if __name__ == '__main__':
    main()
