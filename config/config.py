# --- Jommarn-Omni 231M (Gemma-4 Powered) Configuration ---

# เป้าหมายพารามิเตอร์: ~231 Million
# จุดเด่น: ใช้ Tokenizer ของ Gemma-4 เพื่อการรองรับภาษาไทยระดับเทพ

# ตัวเลข VOCAB_SIZE ของ Gemma ปกติคือ 256,000 
# เราจะตั้งค่าเผื่อให้หารด้วย 64 ลงตัวเพื่อประสิทธิภาพ GPU (256000 + padding)
import os
import torch

VOCAB_SIZE = 152064         # Typhoon OCR Vocab
CONTEXT_LENGTH = int(os.getenv("CONTEXT_LENGTH", "512"))       
N_EMBED = 768               
N_HEAD = 12                  
N_BLOCKS = 32               # 32 Layers
N_KV_HEADS = 2              # 2 KV Heads (GQA 6:1 Ratio)
V_LAYERS = 16               # 16 Vision Layers

# Paths
TRAIN_PATH = "data/train/pile_train.h5"
DEV_PATH = "data/val/pile_dev.h5"
TOKENIZER_PATH = "tokenizer.json"

# Training parameters (Optimized for Kaggle Dual Tesla T4 GPUs 2x15GB with MTP Distillation)
T_BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))           # 1 sequence per GPU to prevent CUDA OOM during MTP+Distillation
T_GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "64"))          # Accumulate 64 steps (Effective batch size = 128)
T_CONTEXT_LENGTH = CONTEXT_LENGTH     
T_TRAIN_STEPS = 100000     
T_EVAL_STEPS = 50         
T_EVAL_ITERS = 100         
T_LR_DECAY_STEP = 20000    
T_LR = 1e-4                 
T_LR_DECAYED = 2e-5        
T_OUT_PATH = "models/jommarn_omni_206m_l40s.pt"

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

default_config = {
    'vocab_size': VOCAB_SIZE,
    'context_length': CONTEXT_LENGTH,
    'n_embed': N_EMBED,
    'n_head': N_HEAD,
    'n_blocks': N_BLOCKS,
    'n_kv_heads': N_KV_HEADS,
    'train_path': TRAIN_PATH,
    'dev_path': DEV_PATH,
    'tokenizer_path': TOKENIZER_PATH,
    'v_layers': V_LAYERS,
    't_batch_size': T_BATCH_SIZE,
    't_grad_accum': T_GRAD_ACCUM, # เพิ่มเข้าไปเพื่อให้โค้ดเรียกใช้ได้
    't_context_length': T_CONTEXT_LENGTH,
    't_train_steps': T_TRAIN_STEPS,
    't_eval_steps': T_EVAL_STEPS,
    't_eval_iters': T_EVAL_ITERS,
    't_lr_decay_step': T_LR_DECAY_STEP,
    't_lr': T_LR,
    't_lr_decayed': T_LR_DECAYED,        # ✅ 2e-5 (ค่า LR หลัง decay)
    't_out_path': T_OUT_PATH,
    'device': DEVICE,
}