# สรุปสถาปัตยกรรมและการพัฒนา "Jommarn-Omni ~501M" (Multimodal & 4-Token MTP Evolution)

> **"Intelligence Density ในขนาดกระทัดรัด: จอมมารผู้มองเห็น คิดก่อนตอบ และซิ่งแรง 4 เท่าด้วย MTP"**

---

## 🌟 จุดเด่นของ Jommarn-Omni ~501M

*   **Native Multimodal Early Fusion:** รวมการประมวลผลข้อมูลภาพ (Vision Encoder) และข้อความ (Text Thinker) เข้าสู่พื้นที่ความหมายเดียวกันตั้งแต่ต้น โดยการนำ Vision Tokens วางไว้หน้า Text Tokens (Prepend) ช่วยให้ตัวแบบเข้าใจภาพและข้อความได้อย่างสัมพันธ์กันอย่างลึกซึ้ง
*   **4-Token Multi-Token Prediction (MTP):** สถาปัตยกรรมปฏิวัติวงการ (อ้างอิงแนวคิดจาก DeepMind/Meta) ใช้ MTP Mixers 3 ตัวเรียนรู้และทำนายโทเคนล่วงหน้า $t+2, t+3, t+4$ ไปพร้อมๆ กับหัวทำนายหลัก ($t+1$) ในขั้นตอนการเทรน และช่วยเร่งความเร็วในการเจนคำตอบในขั้นตอน Inference ขึ้นถึง 3-4 เท่าด้วยการเดาคำล่วงหน้าแบบขนาน
*   **Grouped-Query Attention (GQA):** ปรับปรุงกลไก Attention ให้รองรับสัดส่วน 12 Query headsต่อ 2 KV heads (GQA 6:1 Ratio) ช่วยประหยัดหน่วยความจำ VRAM สำหรับจัดเก็บ KV Cache ทำให้สามารถรันบนบริบทข้อความยาวๆ ได้โดยไม่เปลืองทรัพยากร
*   **16-Layer Vision Encoder:** ขยายระดับชั้นการมองเห็นเป็น 16 เลเยอร์ (เทียบเท่ามาตรฐาน ViT-Large) เพื่อรองรับการสกัดคุณลักษณะ (Features) ของภาพถ่าย เอกสาร หน้าเว็บ และลายมือภาษาไทยขึ้นมาจากศูนย์ (Train from scratch) อย่างละเอียด
*   **Typhoon/Qwen Tokenizer (152k Vocab):** ปรับใช้ Tokenizer ที่จูนภาษาไทยมาเป็นอย่างดี (จาก Qwen2.5-VL/Typhoon) แทนการใช้ Vocab ขนาดใหญ่เกินความจำเป็น ช่วยลดน้ำหนักพารามิเตอร์ของ Embedding ลงกว่า 80 ล้านพารามิเตอร์ และช่วยให้ตัวแบบเข้าใจไวยากรณ์ไทยได้เร็วและประหยัดโทเคนขึ้น
*   **Weight Tying:** มีการแชร์น้ำหนักพารามิเตอร์ระหว่าง Token Embedding และ LM Head (`self.token_embed.weight = self.lm_head.weight`) เพื่อลดขนาดของไฟล์โมเดลและป้องกันปัญหาความไม่สอดคล้องของเวกเตอร์ภาษา

---

## 🔬 รายละเอียดสถาปัตยกรรมเชิงลึก (Architecture Specs)

### 1. องค์ประกอบตัวโมเดลหลัก (Shared Trunk)
*   **Total Parameters:** ~501 ล้านพารามิเตอร์ (หากนับน้ำหนักทั้งหมดรวมถึงส่วนที่แชร์และโมดูลการมองเห็น) หรือประมาณ 384 ล้านพารามิเตอร์ที่ไม่นับ Embedding
*   **Layers (N_BLOCKS):** 32 Blocks (เพิ่มความลึกระดับเดียวกับ LLaMA/Mistral เพื่อความสามารถในการคิดเชิงตรรกะที่ซับซ้อนขึ้น)
*   **Embedding Dimension (N_EMBED):** 768
*   **Attention Heads:** 12 Query Heads (Head Size = 64)
*   **KV Heads (GQA):** 2 KV Heads (แชร์ Key-Value Cache ร่วมกันในอัตราส่วน 6:1)
*   **Context Length:** 4,096 โทเคน
*   **Activation Function:** SwiGLU MLP (ขยายมิติ 4 เท่าไปยัง 3,072 มิติ เพื่อเพิ่มความจุทางความรู้)
*   **Normalization:** RMSNorm
*   **Positional Embedding:** p-RoPE (Partial Rotary Position Embeddings) สำหรับระบุตำแหน่งและตำแหน่งเชิงพื้นที่ของภาพ

### 2. โมดูลการมองเห็น (Jommarn-Vision Encoder)
*   **Layers:** 16 Blocks (Vision Block ประกอบด้วย RMSNorm + nn.MultiheadAttention + SwiGLU MLP)
*   **Image Resolution:** 512x512 พิกเซล (แปลงภาพขนาดใหญ่และเอกสารออกเป็น 1,024 Vision Tokens โดยใช้ Patch Size ขนาด 16x16 พิกเซล) เพื่อให้อ่านฟอนต์ตัวหนังสือขนาดเล็กบนหน้าบิลได้คมชัด
*   **Parameter Size:** ~114M พารามิเตอร์ (เฉพาะโมดูล Vision Block และ Patch Embeddings)

### 3. ระบบ Multi-Token Prediction (MTP Mixers)
*   **Mixers:** 3 ชุดขนานทำงานร่วมกัน (MTP1, MTP2, MTP3)
*   **การทำงาน:** นำ Hidden State ชั้นสุดท้ายของ Transformer มาเชื่อมต่อ (Concatenate) กับเวกเตอร์คำตอบของตัวก่อนหน้า แล้วผ่าน Linear layer + RMSNorm + GELU + Linear layer เพื่อทำนายคำถัดๆ ไปขนานกัน
*   **Parameter Size:** ~7M พารามิเตอร์

---

## 📚 ชุดข้อมูลการเรียนรู้ (6-Way Balanced Dataset + Distillation)

เราจัดสัดส่วนการสุ่มเรียนรู้ (Sampling Weight) ไว้เพื่อดึง Loss ให้ลงเร็วและเสถียรที่สุดดังนี้:

1.  **Distillation Warmup Data:** 40% (ข้อมูลที่ประมวลผลล่วงหน้าโดยใช้ `Typhoon-OCR-7B` ช่วยตอบ เพื่อสอนให้โมเดลเริ่มอ่านอักษรและจับใจความได้อย่างรวดเร็ว)
2.  **Detailed OCR & CoT (`Phonsiri/handwrite-ocr-detailed`):** 12% (อธิบายรายละเอียดรูปภาพและการคิดตามลำดับ)
3.  **Astrology & Document Layout (`Phonsiri/astrology-dataset-clean`):** 12% (เอกสารและอินโฟกราฟิกด้านโหราศาสตร์)
4.  **COCO Thai General Understanding (`Phonsiri/coco-thai-gemma4-detailed`):** 12% (ความเข้าใจภาพและวัตถุทั่วไปในไทย)
5.  **Handwriting OCR (`iapp/thai_handwriting_dataset`):** 12% (ลายมือไทยจริง)
6.  **Thai Wikipedia (`pythainlp/thai-wiki-dataset-v3`):** 12% (รักษารากฐานข้อมูลภาษาไทยทั่วไป)

---

## 📉 สถานะการเทรนล่าสุด (Training Progress)

การเทรนโมเดลได้รันบนระบบ Cloud GPU ที่ใช้การ์ดจอประสิทธิภาพสูง และถูกบันทึกสถานะล่าสุดไว้ดังนี้:

*   **เครื่องมือรัน:** AMD MI300X 192GB VRAM GPU (เช่าใช้ผ่าน DigitalOcean)
*   **พารามิเตอร์การคุม:** Batch size 32, Gradient Accumulation 16 (ทำให้ได้ Effective Batch Size = 512)
*   **อัตราการเรียนรู้ (Learning Rate):** เริ่มต้นที่ 1e-4 พร้อม Warmup 2,000 สเต็ป และค่อยๆ ทำ Cosine decay ลงไปที่ 2e-5 ที่สเต็ป 20,000
*   **ความคืบหน้าล่าสุด:**
    *   **Step ล่าสุด:** **14,983** จากสเต็ปตั้งเป้าสูงสุด 85,911 (คิดเป็นประมาณ 17.4% ของการเทรนทั้งหมด)
    *   **Loss ล่าสุด:** ลดลงมาอยู่ที่ประมาณ **~1.90** (จากจุดเริ่มต้นที่ Loss สูงกว่า 10.0)
    *   **จำนวนโทเคนที่เรียนรู้ไปแล้ว (Tokens Trained):** ประมาณ **~1.9 พันล้านโทเคน (1.9 Billion Tokens)**
    *   **เวลาการทำงานสะสม:** ~48.5 ชั่วโมง
*   **การสำรองข้อมูล (Checkpoints):**
    *   อัปโหลดและสำรองข้อมูลไปยัง Hugging Face Hub เรียบร้อยแล้วที่: [Phonsiri/jommarn-omni-checkpoints](https://huggingface.co/Phonsiri/jommarn-omni-checkpoints)
    *   ชื่อไฟล์ Checkpoint ล่าสุด: `jommarn_omni_206m_l40s_latest.pt` (ขนาดไฟล์ประมาณ 9.02 GB ซึ่งรวมค่า Optimizer states และ Scheduler states ไว้สำหรับการกด Resume ได้ทันที)

---

## 👁️ ผลการทดสอบและข้อสังเกตเชิงพฤติกรรม (Empirical Observations)

เมื่อนำโมเดลที่ Step ~14,983 มาทดสอบตอบคำถามผ่านตัวสคริปต์ `scripts/test_omni.py` ด้วยพรอมต์ภาษาไทย เช่น "ประเทศไทย" และ "การดำเนินการ" มีผลลัพธ์และพฤติกรรมที่น่าสนใจดังนี้:

### 1. โครงสร้างและการจัดรูปแบบภาษาเริ่มพัฒนา (Emerging Language Structure)
*   โมเดลสามารถเข้าใจหลักไวยากรณ์ คำนาม คำกริยาของภาษาไทยได้อย่างเป็นธรรมชาติ
*   เริ่มมีพฤติกรรมการจัดหน้าในลักษณะของสารานุกรมวิกิพีเดีย เช่น การสร้างหัวข้อ `== อ้างอิง ==` หรือการแบ่งรายการหัวข้อย่อย

### 2. การทำงานของระบบตรวจจับคำสิ้นสุด (EOS Detection)
*   โมเดลสามารถสั่งหยุดการทำงานของตัวเองได้เมื่อเจนเนื้อหาจนจบความต้องการจริง โดยสามารถตรวจพบ EOS token (`[EOS detected at step ...]`) เพื่อยุติการทำงานได้อย่างถูกต้อง

### 3. พฤติกรรมแบบโมเดลข้อความพื้นฐาน (Base Model Autocomplete Behavior)
*   เนื่องจากโมเดลเพิ่งผ่านการเทรนขั้น Pre-training ไปเพียง 17% และยังไม่ได้เข้าสู่สเตจ **SFT (Supervised Fine-Tuning)** หรือการสอนทำตามคำสั่ง (Instruction Tuning) โมเดลจึงยังทำงานเป็นตัวเติมคำถัดไปตามความน่าจะเป็น (Autocomplete) มากกว่าที่จะตอบคำถามแบบ Chatbot เช่น:
    *   พรอมต์: *"การดำเนินการ"*
    *   ผลลัพธ์: *"น้ำร้อนแล้ว\nป้องหมูหองครับบรพระคั่ล่อแผ่น..."* (แต่งคำตามความคุ้นเคยของคำในฐานข้อมูล แต่ยังไม่มีตรรกะคำถาม-คำตอบ)
    *   พรอมต์: *"ประเทศไทย"*
    *   ผลลัพธ์: *"วันนัแพทย์ซิฉูปกลิ๊บรับขุดระย้ายจบการความสบาย..."* (ปะติดปะต่อคำศัพท์ภาษาไทยที่ถูกต้อง แต่เนื้อหาโดยรวมยังไม่มีความหมายเชื่อมโยงกัน)
*   พบเจออาการเมาค้างของโทเคน (Degeneration loops) เป็นบางครั้ง เช่น การวนซ้ำตัวอักษรบางตัวหรือการวนลูปวลี ซึ่งเป็นพฤติกรรมปกติของโมเดลที่ยังเทรนไม่จบ Phase และยังขาดการฟินจูน

---

## 🔬 การทดลองเพิ่มเติม: โครงสร้าง VLM แบบไร้ Encoder (Encoder-Free Experiment)

ในเครื่องหรือการรันจำลองขนาดจิ๋ว (Toy VLM) เราได้พัฒนาระบบตัวแบบจำลองที่ลดความซับซ้อนใน `experiments/toy_vlm_experiment.py`:

*   **แนวคิด (Encoder-Free Patching):** เพื่อลดภาระการประมวลผลและการใช้ VRAM มหาศาลในการรันโครงสร้างตา (Vision Transformer ขนาดใหญ่) เราตัด Convolution และ ViT Blocks ทั้งหมดในส่วนของ Vision ออก แล้วใช้การคำนวณแบบหั่นภาพพิกเซลดิบโดยตรง (อิงแนวคิดแนวทางของ Gemma 4 12B)
*   **การประมวลผลภาพ:**
    1.  หั่นรูปภาพขนาด $64 \times 128$ พิกเซล ออกเป็นชิ้นส่วน (Patches) ขนาด $16 \times 16$ พิกเซล (ได้ชิ้นส่วนภาพทั้งหมด 32 ชิ้น)
    2.  นำช่องสีและพิกเซลมาทุบรวมกันเป็นเวกเตอร์ดิบขนาด $3 \text{ channels} \times 16 \times 16 = 768$ มิติ
    3.  โยนเข้าผ่านเลเยอร์เส้นตรง `nn.Linear` (Projection Layer) ตัวเดียวแปลงจาก 768 มิติเข้าสู่มิติโมเดล `d_model` เพื่อใช้แทนโทเคนภาพทันที
*   **การทดสอบ:** ฝึกฝนบนข้อมูลลายมือภาษาไทย (`iapp/thai_handwriting_dataset`) จำนวน 10 ภาพเป็นเวลา 500 Epochs ผลลัพธ์แสดงให้เห็นว่าตัวแบบจิ๋วสามารถจำลายมือตัวเขียนภาษาไทยที่กำหนดได้เป็นอย่างดีเมื่อติดตั้งกลไก **Repetition Penalty** และ **Temperature Sampling** เพื่อป้องกันอาการทายคำซ้ำซาก (Mode Collapse)

---

## 🏃‍♂️ คู่มือการใช้งานและจัดการโมเดลขั้นสูง

### 1. การกู้คืนการเทรน (Resuming Training)
เมื่อมีการเปิดเครื่อง Cloud GPU ใหม่ สามารถดึงไฟล์ Checkpoint ล่าสุดจาก Hugging Face ลงเครื่องและรันต่อโดยอัตโนมัติด้วยคำสั่ง:
```bash
export HF_REPO_ID="Phonsiri/jommarn-omni-checkpoints"
python scripts/train_transformer.py
```
*(ระบบจะตรวจหาไฟล์ `jommarn_omni_206m_l40s_latest.pt` บน Hub แล้วดาวน์โหลดมาคลี่เอา Model, Optimizer, และ Scheduler state เพื่อรันต่อสเต็ปถัดไปได้ทันที)*

### 2. แผนการรันสลับระยะ (Phase Management)
*   **ระยะที่ 1: พูดไทยให้เป๊ะ (Text-Only Pre-training):**
    ```bash
    TRAIN_PHASE="text_only" FORCE_RESET=1 HF_REPO_ID="Phonsiri/jommarn-omni-checkpoints" python scripts/train_transformer.py
    ```
*   **ระยะที่ 2: ผนวกดวงตา (Multimodal Early Fusion):**
    เมื่อ Loss ภาษาไทยนิ่งดีแล้ว ให้สั่งเทรนระยะ 2 โดยลบ `FORCE_RESET` ออกเพื่อให้โหลดค่าเดิมมารันต่อ:
    ```bash
    TRAIN_PHASE="multimodal" HF_REPO_ID="Phonsiri/jommarn-omni-checkpoints" python scripts/train_transformer.py
    ```

### 3. การรันรวดเร็วสำหรับการทดสอบ (Inference via Terminal)
สั่งรันสคริปต์ทดสอบพรอมต์และรูปภาพได้ผ่านคอมมานด์ไลน์:
```bash
python scripts/test_omni.py --prompt "ประเทศไทย" --temperature 0.7 --repetition_penalty 1.2
```

---

## 🛠️ วิธีการจำกัดพื้นที่ดิสก์และประหยัด VRAM บน Cloud
1.  **ลบแคชดาวน์โหลดขยะของ HF:** เมื่อเทรนไปนานๆ Hugging Face จะสร้าง Cache ไฟล์ขยะสะสม ให้ขยันสั่งรันบรรทัดนี้:
    ```bash
    rm -rf ~/.cache/huggingface/datasets/
    rm -rf ~/.cache/huggingface/hub/
    ```
2.  **ลบไฟล์สเต็ปเก่า:** สคริปต์จะเก็บ Checkpoint ล่าสุดไว้ 2 ตัวโดยอัตโนมัติ แต่อาจมีไฟล์ตกค้าง สามารถล้างได้โดย:
    ```bash
    rm -f models/*_step_*.pt
    ```

---

## 🔮 แผนงานในอนาคต (Next Steps Roadmap)
1.  **ขยายเวลาเครื่องรัน:** ดำเนินการตอบรับและยื่นตั๋วสิทธิ์ใช้งาน AMD GPU Credits บน Digital Ocean ต่อเนื่องเพื่อให้ได้ระบบ MI300X กลับมาเทรนให้ผ่านสเต็ป 50,000+
2.  **SFT Phase (Supervised Fine-Tuning):** เตรียมชุดคำสั่งถามตอบภาษาไทย (Instruct Dataset) เพื่อปรับพฤติกรรมโมเดลจาก Autocomplete มาเป็นผู้ช่วยตอบคำถามที่แสนฉลาด
3.  **RL & GRPO Alignment:** จูนตรรกะและความปลอดภัยของโมเดลผ่านระบบ Policy Optimization ในขั้นตอนสุดท้าย

---
*บันทึกวิวัฒนาการโครงการ Jommarn-Omni*
😈📸⚡
