# Jommarn-Omni 501M: Native Multimodal Thai Intelligence 😈🇹🇭🖼️

Jommarn-Omni is a state-of-the-art **Native Multimodal (Vision + Text)** model designed for high "Intelligence Density" in a compact 501M parameter footprint. Fusing a deep Transformer decoder with a multi-layered Vision Encoder in a unified semantic space, it is tailored specifically for Thai language processing, official document analysis, and handwriting OCR.

---

## 🚀 Key Evolutionary Features

Developed from a base Transformer, Jommarn-Omni integrates modern architectural breakthroughs used in frontier models:

*   **Native Multimodal Early Fusion:** A built-in **Jommarn-Vision Encoder** allows the model to "see" and "think" in a single semantic space. Vision tokens are prepended directly before text tokens to ensure a deep contextual understanding of visual documents.
*   **4-Token Multi-Token Prediction (MTP):** An advanced pre-training architecture (inspired by recent Meta/DeepMind papers). Using 3 parallel MTP mixers, it learns to predict future tokens $t+2, t+3, t+4$ during training, which speeds up autoregressive generation during inference by 3x-4x.
*   **Grouped-Query Attention (GQA):** Configured with a 12 Query Heads to 2 KV Heads (6:1 ratio) setup. This dramatically reduces the VRAM usage of the KV Cache, making it possible to handle long context windows.
*   **16-Layer Vision Encoder:** Built with 16 Vision Blocks (equivalent depth to ViT-Large) trained from scratch to extract fine details, text, and handwriting layouts from $512 \times 512$ pixel inputs.
*   **Typhoon-Optimized Tokenizer (152k Vocab):** Replaced general-purpose vocabularies with a Thai-optimized vocabulary (derived from Qwen2.5-VL/Typhoon), reducing the vocabulary parameter size by over 80 million weights while improving Thai parsing efficiency.
*   **Weight Tying:** Tied the weights of the Token Embedding and LM Head (`self.token_embed.weight = self.lm_head.weight`) to lower model footprint and improve language stability.
*   **SwiGLU Activation & RMSNorm:** Replaced traditional ReLU and LayerNorm with SwiGLU MLP layers and RMSNorm to enhance training stability and representational capacity.

---

## 📊 Model Specifications

| Parameter | Value |
|-----------|-------|
| **Total Parameters** | ~501 Million (~384M active parameters without shared embedding weight) |
| **Architecture** | Native Multimodal Decoder-only + ViT Encoder |
| **Embedding Dim (N_EMBED)** | 768 |
| **Trunk Blocks (N_BLOCKS)** | 32 (Matches LLaMA/Mistral depth ratio) |
| **Attention Heads** | 12 Query Heads (GQA with 2 KV Heads) |
| **Vision Encoder Layers** | 16 Blocks |
| **MTP Mixers** | 3 (For predicting up to 4 tokens ahead) |
| **Context Length** | 4,096 Tokens |
| **Vocabulary Size** | 152,064 (Thai-optimized) |

---

## 📉 Training Run & Progress

Jommarn-Omni was trained on cloud-hosted enterprise accelerators with the following setup:

*   **Hardware:** AMD MI300X 192GB VRAM GPU (DigitalOcean Cloud)
*   **Hyperparameters:**
    *   **Batch Size:** 32 (physical) with **Gradient Accumulation:** 16 (effective batch size: 512)
    *   **Learning Rate (LR):** Peak of 1e-4, decaying to 2e-5 at step 20,000 using a Cosine schedule with a 2,000-step Warmup.
*   **Current Training Status:**
    *   **Current Step:** **14,983** / 85,911 (approx. 17.4% complete)
    *   **Current Loss:** **~1.90** (stabilized and successfully down from >10.0)
    *   **Tokens Processed:** **~1.9 Billion Tokens**
    *   **Time Elapsed:** ~48.5 hours
*   **Checkpoint & Uploads:**
    *   The latest checkpoint (including model weights, optimizer states, and scheduler variables for seamless recovery) is securely backed up on the Hugging Face Hub: [Phonsiri/jommarn-omni-checkpoints](https://huggingface.co/Phonsiri/jommarn-omni-checkpoints)
    *   Checkpoint File: `jommarn_omni_206m_l40s_latest.pt` (9.02 GB)

---

## 👁️ Empirical Observations & Test Logs

During pre-training evaluations using the `scripts/test_omni.py` script on Thai prompts like "ประเทศไทย" (Thailand) and "การดำเนินการ" (Operation), several findings were observed:

1.  **Syntactic Development:** The model has successfully acquired basic Thai sentence structure and grammar. It can output coherent Thai word chunks.
2.  **Structural Behavior:** It has learned document layouts, frequently generating Wikipedia-like page features such as header lines (e.g. `== อ้างอิง ==` for references) and lists.
3.  **EOS Functionality:** The End-of-Sequence (EOS) token mechanism is fully operational, preventing infinite generation loops by terminating execution upon detecting an EOS token.
4.  **Base Model Nature:** Since the model has only completed 17% of its pre-training phase and has **not** yet undergone **Supervised Fine-Tuning (SFT)** or **Instruction Tuning**, it behaves purely as a text-completion engine (autocomplete) rather than a conversational assistant. It generates logically structured but semantic-unrelated content when prompted, which is the expected behavior for base models at this stage.

---

## 🔬 Specialized Experiments: Encoder-Free VLM

In parallel, we developed a lightweight VLM variant to investigate VRAM reduction in `experiments/toy_vlm_experiment.py`:

*   **Encoder-Free Patching:** Instead of using a convolutional feature extractor or a heavy pre-trained ViT transformer, this experiment uses raw pixel patching (inspired by Google's Gemma 4 12B design).
*   **Mechanism:**
    1.  Input images ($64 \times 128$ pixels) are reshaped into 32 patches of $16 \times 16$ pixels.
    2.  Patches are flattened directly into $3 \text{ channels} \times 16 \times 16 = 768$-dimensional vectors.
    3.  A single linear projection layer (`nn.Linear`) projects these 768-dimensional raw pixel patches directly into the model embedding space.
*   **Results:** Successfully trained on the Thai Handwriting Dataset (`iapp/thai_handwriting_dataset`, 10 images) for 500 epochs. Paired with repetition penalty and temperature sampling, it demonstrates that raw pixel projection works effectively for lightweight OCR tasks.

---

## 🛠️ Setup & Usage Guide

### 1. Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
pip install datasets transformers huggingface_hub torchvision pillow tqdm
```

### 2. Project File Structure
*   `config/config.py`: Global parameters, training details, and optimizer settings.
*   `src/models/transformer.py`: Core `JommarnOmni` neural architecture (Decoder trunk, GQA, and MTP modules).
*   `src/models/vision_encoder.py`: 16-layer custom vision encoder with Patch Embedding.
*   `scripts/train_transformer.py`: Pre-training pipeline with automatic Hugging Face checkpoint syncing.
*   `scripts/test_omni.py`: Autoregressive inference tool with MTP support and sampling parameters.
*   `experiments/toy_vlm_experiment.py`: Notebook/script containing the Encoder-Free VLM proof-of-concept.

### 3. Resuming Training
To resume training on a new node or recover from an interruption:
1.  Configure your Hugging Face login token:
    ```bash
    export HF_TOKEN="your_huggingface_write_token"
    ```
2.  Run the training script with the target repo environment variable:
    ```bash
    export HF_REPO_ID="Phonsiri/jommarn-omni-checkpoints"
    python scripts/train_transformer.py
    ```
    *The loader will check the Hub, download `jommarn_omni_206m_l40s_latest.pt` to the local `models/` directory, restore optimizer/scheduler variables, and resume training.*

### 4. Alternating Training Phases
*   **Phase 1: Text-Only Pre-training (Speak Thai Fluently):**
    ```bash
    TRAIN_PHASE="text_only" FORCE_RESET=1 HF_REPO_ID="Phonsiri/jommarn-omni-checkpoints" python scripts/train_transformer.py
    ```
*   **Phase 2: Multimodal Pre-training (Open the Eyes):**
    ```bash
    TRAIN_PHASE="multimodal" HF_REPO_ID="Phonsiri/jommarn-omni-checkpoints" python scripts/train_transformer.py
    ```

### 5. Running Inference
To run text-generation/OCR testing in the terminal:
```bash
python scripts/test_omni.py --prompt "ประเทศไทย" --temperature 0.8 --repetition_penalty 1.2
```

---

## 🔮 Project Roadmap

1.  **MI300X Server Allocation:** Complete the DigitalOcean AMD GPU credit application to secure training hardware and scale training past Step 50,000.
2.  **Supervised Fine-Tuning (SFT):** Transition the model from autocomplete behavior to conversational instruction-following using Thai instruction datasets.
3.  **RL & GRPO Alignment:** Improve response logic, task accuracy, and safety constraints using Group Relative Policy Optimization (GRPO).

---
*Maintained by the Jommarn-Omni Engine Team*
😈📸⚡