# %% [markdown]
# # Toy Vision-Language Model (VLM) for Thai Captioning
# จำลองสถาปัตยกรรม VLM จิ๋ว โดยใช้ภาพและแคปชั่นภาษาไทยจริงๆ จาก Hugging Face (10 ภาพ)

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from torchvision import transforms

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ## 2. โหลดข้อมูลจริงจาก Hugging Face (เอาแค่ 10 ภาพ)
# %%
print("Downloading dataset...")
# เปลี่ยนมาใช้ Dataset ลายมือภาษาไทย (iapp/thai_handwriting_dataset)
dataset = load_dataset("iapp/thai_handwriting_dataset", split="train[:10]")

# ลายมือมักจะเป็นสี่เหลี่ยมผืนผ้าแนวยาว เราจะเปลี่ยนเป็น 64x128
transform = transforms.Compose([
    transforms.Resize((64, 128)), # ปรับให้ภาพยาวขึ้นเพื่อรับลายมือ
    transforms.ToTensor()
])

real_images = []
real_texts = []

for item in dataset:
    img = transform(item["image"].convert("RGB"))
    text = item.get("text", "ไม่มีข้อความ") # คอลัมน์ข้อความของ iapp คือ 'text'
    
    real_images.append(img)
    real_texts.append(text)

print(f"Loaded {len(real_images)} images and captions.")
print(f"Example caption: {real_texts[0]}")

# %% [markdown]
# ## 3. สร้าง Tokenizer (Character-level สำหรับภาษาไทย)
# %%
class CharTokenizer:
    def __init__(self, corpus):
        chars = set("".join(corpus))
        self.vocab = ["<pad>", "<bos>", "<eos>"] + list(chars)
        self.char_to_id = {c: i for i, c in enumerate(self.vocab)}
        self.id_to_char = {i: c for i, c in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        
    def encode(self, text, max_len=512): # เพิ่มความยาวกระเป๋าเป็น 512 ตัวอักษร
        tokens = [self.char_to_id["<bos>"]]
        tokens += [self.char_to_id[c] for c in text if c in self.char_to_id]
        tokens += [self.char_to_id["<eos>"]]
        if len(tokens) < max_len:
            tokens += [self.char_to_id["<pad>"]] * (max_len - len(tokens))
        return torch.tensor(tokens[:max_len], dtype=torch.long)
        
    def decode(self, token_ids):
        out = ""
        for i in token_ids:
            if i == self.char_to_id["<pad>"]: continue
            if i == self.char_to_id["<bos>"]: continue
            if i == self.char_to_id["<eos>"]: break
            out += self.id_to_char[int(i)]
        return out

tokenizer = CharTokenizer(real_texts)
print(f"Vocab Size: {tokenizer.vocab_size} Thai/Eng Characters")

X_img = torch.stack(real_images) # เอา .to(device) ออก ปล่อยไว้ที่ CPU RAM ก่อน
Y_tokens = torch.stack([tokenizer.encode(t) for t in real_texts]) # เอา .to(device) ออก

print(f"X_img shape: {X_img.shape}")
print(f"Y_tokens shape: {Y_tokens.shape}")

# %% [markdown]
# ## 4. สร้าง Toy VLM
# %%
class EncoderFreePatching(nn.Module):
    def __init__(self, d_model, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        # 🚫 ไม่มี Convolution 🚫 ไม่มี Encoder หนาๆ
        # เอาพิกเซลดิบๆ 16x16 pixels * 3 สี = 768 ค่า โยนเข้าสมอง LLM เลย
        self.raw_pixel_proj = nn.Linear(3 * patch_size * patch_size, d_model)
        
    def forward(self, x):
        # x คือภาพต้นฉบับขนาด [Batch, 3, H=64, W=128]
        B, C, H, W = x.shape
        P = self.patch_size
        
        # 1. หั่นภาพเป็นชิ้นๆ (Patching) โดยไม่ใช้ Convolution (ใช้แค่การจัดเรียง Array)
        # แปลง [B, 3, 64, 128] -> [B, 3, 4, 16, 8, 16]
        patches = x.view(B, C, H // P, P, W // P, P)
        
        # 2. จัดเรียงใหม่ให้เป็นลำดับชิ้นภาพ
        # -> [B, แถว, คอลัมน์, สี, P, P] -> [B, 4, 8, 3, 16, 16]
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        
        # 3. ทุบให้แบนราบ (Flatten)
        # -> [B, 32 ชิ้น, 768 ค่าพิกเซล]
        patches = patches.view(B, -1, C * P * P)
        
        # 4. วุ้นแปลภาษา: โยนพิกเซลดิบ 768 ค่า เข้า LLM โดยตรงเลย! (Encoder-free!)
        return self.raw_pixel_proj(patches)


class ToyVLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_layers=2, n_heads=4):
        super().__init__()
        self.vision_encoder = EncoderFreePatching(d_model)
        self.word_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(600, d_model) # เพิ่มสมองให้จำตำแหน่งได้ 600 ตัวอักษร
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, images, text_tokens):
        B, seq_len = text_tokens.shape
        img_features = self.vision_encoder(images)
        positions = torch.arange(seq_len, device=text_tokens.device).unsqueeze(0).expand(B, -1)
        text_features = self.word_embedding(text_tokens) + self.pos_embedding(positions)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(text_tokens.device)
        out = self.transformer(tgt=text_features, memory=img_features, tgt_mask=mask)
        return self.lm_head(out)

model = ToyVLM(vocab_size=tokenizer.vocab_size).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

# %% [markdown]
# ## 5. Training Loop
# %%
from torch.utils.data import TensorDataset, DataLoader

optimizer = optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.char_to_id["<pad>"])

epochs = 500
loss_history = []

# ใช้ DataLoader เพื่อจัดการ Mini-batch
batch_size = 32
train_dataset = TensorDataset(X_img, Y_tokens)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print("🚀 Starting Training...")
for epoch in range(epochs):
    epoch_loss = 0
    for images, tokens in train_loader:
        # ย้ายข้อมูลเข้า GPU เฉพาะ Batch นี้
        images = images.to(device)
        tokens = tokens.to(device)
        
        optimizer.zero_grad()
        input_ids = tokens[:, :-1]
        target_ids = tokens[:, 1:]
        
        logits = model(images, input_ids)
        loss = criterion(logits.reshape(-1, tokenizer.vocab_size), target_ids.reshape(-1))
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    avg_epoch_loss = epoch_loss / len(train_loader)
    loss_history.append(avg_epoch_loss)
    
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_epoch_loss:.4f}")

plt.plot(loss_history)
plt.title("Training Loss (Mini-batch)")
plt.show()

# %% [markdown]
# ## 6. Inference (ทดสอบอ่านภาพ)
# %%
# ฟังก์ชัน Sample แทน argmax เพื่อแก้ปัญหา Mode Collapse
def generate(model, img_tensor, tokenizer, max_steps=100, temperature=1.2, rep_penalty=1.5):
    model.eval()
    generated_tokens = [tokenizer.char_to_id["<bos>"]]
    with torch.no_grad():
        for step in range(max_steps):
            input_tensor = torch.tensor([generated_tokens]).to(device)
            logits = model(img_tensor, input_tensor)
            
            # ดึง logits ตัวสุดท้ายออกมา
            last_logits = logits[0, -1, :].float()
            
            # Repetition Penalty: ลงโทษตัวที่พิมพ์ไปแล้วให้มีโอกาสถูกเลือกน้อยลง
            for prev_token in set(generated_tokens):
                last_logits[prev_token] /= rep_penalty
            
            # Temperature Sampling: เพิ่มความหลากหลายในการเลือกตัวอักษร
            probs = torch.softmax(last_logits / temperature, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            generated_tokens.append(next_token_id)
            if next_token_id == tokenizer.char_to_id["<eos>"]: break
    return generated_tokens

print("\U0001f9e0 Model is thinking...")

# ทดสอบรูปที่ 1
test_idx = 0
test_img = X_img[test_idx:test_idx+1].to(device)
result = generate(model, test_img, tokenizer)
print(f"\n✅ Expected (เฉลย): {real_texts[test_idx]}")
print(f"\U0001f916 Predicted (โมเดลทาย): {tokenizer.decode(result)}")

# ทดสอบรูปที่ 2
test_idx = 2
test_img = X_img[test_idx:test_idx+1].to(device)
result = generate(model, test_img, tokenizer)
print(f"\n✅ Expected (เฉลยรูป 2): {real_texts[test_idx]}")
print(f"\U0001f916 Predicted (โมเดลทายรูป 2): {tokenizer.decode(result)}")

# %% [markdown]
# ## 7. ทดสอบกับภาพของคุณเอง (เฉพาะบน Google Colab)
# รันเซลล์นี้เพื่ออัปโหลดรูปภาพจากเครื่องของคุณเองให้โมเดลลองอ่าน
# %%
try:
    from google.colab import files
    from PIL import Image
    import io

    print("📸 กรุณาอัปโหลดรูปภาพของคุณ:")
    uploaded = files.upload()

    if uploaded:
        filename = list(uploaded.keys())[0]
        
        # โหลดภาพและแปลงให้เป็น Tensor (64x64)
        img_data = uploaded[filename]
        custom_img = Image.open(io.BytesIO(img_data)).convert("RGB")
        
        plt.imshow(custom_img)
        plt.title("Your Image")
        plt.axis("off")
        plt.show()
        
        # ใส่ batch dimension [1, 3, 64, 128]
        custom_img_tensor = transform(custom_img).unsqueeze(0).to(device)
        
        print("🧠 Model is thinking about your image...")
        # ใช้ generate() ที่มี Temperature + Repetition Penalty เหมือนบล็อก 6 เลยครับ
        result = generate(model, custom_img_tensor, tokenizer,
                          max_steps=150, temperature=0.8, rep_penalty=1.5)
                
        print(f"\n🤖 Predicted (โมเดลทายภาพของคุณ): {tokenizer.decode(result)}")
    else:
        print("❌ ไม่พบรูปภาพที่อัปโหลด")
except ImportError:
    print("โค้ดส่วนนี้รองรับการอัปโหลดไฟล์เฉพาะบน Google Colab เท่านั้นครับ")
