# --- Jommarn-Omni 231M (Gemma-4 Powered) Configuration ---

# เป้าหมายพารามิเตอร์: ~231 Million
# จุดเด่น: ใช้ Tokenizer ของ Gemma-4 เพื่อการรองรับภาษาไทยระดับเทพ

# ตัวเลข VOCAB_SIZE ของ Gemma ปกติคือ 256,000 
# เราจะตั้งค่าเผื่อให้หารด้วย 64 ลงตัวเพื่อประสิทธิภาพ GPU (256000 + padding)
VOCAB_SIZE = 152064         # ปรับให้เข้ากับ Typhoon OCR (หาร 64 ลงตัวเพื่อ L40S)
CONTEXT_LENGTH = 4096       
N_EMBED = 768               
N_HEAD = 12                  
N_BLOCKS = 32               # เพิ่มสมองเป็น 32 ชั้น (เทียบเท่าความลึกของ LLaMA/Mistral)
N_KV_HEADS = 2              # 2 KV Heads (GQA 6:1 Ratio)
V_LAYERS = 16               # เพิ่มตาเป็น 16 ชั้น (มาตรฐาน ViT-Large)

# Paths
TRAIN_PATH = "data/train/pile_train.h5"
DEV_PATH = "data/val/pile_dev.h5"
TOKENIZER_PATH = "tokenizer.json" # ไฟล์ที่ดึงมาจาก Gemma-4

# Training parameters (Optimized for AMD MI300X 192GB - Maximum Performance)
T_BATCH_SIZE = 32           # ขยับเป็น 32 เพื่อรีดเร้นพลัง VRAM (น่าจะกินประมาณ 80-90%)
T_GRAD_ACCUM = 16           # สะสม 16 รอบ เพื่อรักษา Effective Batch Size ให้เป็น 512 เท่าเดิม
T_CONTEXT_LENGTH = 4096     
T_TRAIN_STEPS = 100000     
T_EVAL_STEPS = 50         
T_EVAL_ITERS = 100         
T_LR_DECAY_STEP = 20000    
T_LR = 1e-4                 # ลดลงเหลือ 1e-4 เพื่อความเสถียรของโมเดล 501M
T_LR_DECAYED = 2e-5        
T_OUT_PATH = "models/jommarn_omni_206m_l40s.pt"

# Device
import torch
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