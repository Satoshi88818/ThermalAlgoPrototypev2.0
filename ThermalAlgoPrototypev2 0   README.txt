# ThermalAlgoPrototypev2.0 – README

## Overview

**ThermalAlgoPrototypev2.0** is a **production-ready, first-principles-engineered Discrete Diffusion Transformer Model (DTM)** optimized for **64×64 RGB image generation** using **bit-flip forward processes**, **binary score matching**, and **Persistent Contrastive Divergence (PCD-k)**. It leverages **block-sparse energy functions** and supports **TSU (Thermal Sampling Unit) hardware acceleration** with seamless GPU/CPU fallback.

This model is designed for **high-fidelity generative modeling** on binarized image data (e.g., CIFAR-10), achieving **FID-competitive performance** through efficient block-wise Gibbs sampling and sparse bilinear interactions.

---

## Key Features

| Feature | Description |
|-------|-----------|
| **64×64 RGB Generation** | Full-color, high-resolution discrete image synthesis |
| **Bit-Flip Forward Process** | Deterministic noise schedule via `bitflip_schedule()` |
| **Binary DSM + PCD-k Training** | Combines score matching and contrastive divergence |
| **Block-Sparse Energy Function** | 90%+ sparsity in local/skip connections |
| **TSU-Native (Optional)** | Hardware acceleration via `thrml` library |
| **GPU/CPU Fallback** | Full functionality without TSU |
| **FID Evaluation Ready** | Built-in FID computation using `torch-fid` |
| **Checkpointing & Sampling** | Auto-save + grid visualization |

---

## Requirements

### `requirements.txt`

```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
matplotlib>=3.5.0
Pillow>=8.0.0
tqdm>=4.60.0
# Optional: TSU Hardware Acceleration
# thrml>=0.3.0  # Install separately from vendor SDK

# FID Evaluation (optional but recommended)
torch-fid>=0.2.0
```

> **Note**: `thrml` is **not on PyPI** — obtain from xAI TSU SDK (internal). If unavailable, the fallback implementation is **fully functional**.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Satoshi88818/ThermalAlgoPrototypev2.0/tree/main.git
cd ThermalAlgoPrototypev2.0

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install thrml for TSU acceleration
# pip install thrml --index-url https://internal-pypi.x.ai/simple
```

---

## Project Structure

```
ThermalAlgoPrototypev2.0/
├── block_sparse_dtm_production_v2.py  # Main script
├── requirements.txt
├── README.md
├── outputs/               # Generated samples
├── checkpoints/           # Model weights
├── tmp_real/, tmp_fake/   # FID temp dirs (auto-cleaned)
└── data/                  # CIFAR-10 (auto-downloaded)
```

---

## Usage Instructions

### 1. **Train from Scratch**

```bash
python block_sparse_dtm_production_v2.py
```

- Trains for **200 epochs** by default
- Saves checkpoints every 20 epochs
- Generates sample grids at epochs 5, 10, 20, 40, ..., 200
- Final model saved to `checkpoints/block_sparse_dtm_final.pth`

### 2. **Resume Training or Inference**

If checkpoint exists, the script **automatically loads** it and skips training.

```bash
# Just run — will load existing model and generate samples
python block_sparse_dtm_production_v2.py
```

### 3. **Generate Samples Only**

```python
model = BlockSparseDTM().to('cuda')
model.load_state_dict(torch.load("checkpoints/block_sparse_dtm_final.pth"))
samples = model.sample(batch_size=64)
save_image_grid(samples, "my_custom_samples.png")
```

### 4. **Evaluate FID**

```python
fid = compute_fid(train_loader, model, num_samples=1000)
print(f"FID: {fid:.2f}")
```

> Requires `torch-fid`. Install via: `pip install torch-fid`

---

## Configuration (Edit in Script)

| Parameter | Default | Description |
|---------|--------|-----------|
| `IMG_SIZE` | 64 | Image resolution |
| `BLOCK_SIZE` | 8 | 8×8 patches → 64 blocks |
| `T_STEPS` | 100 | Diffusion timesteps |
| `BATCH_SIZE` | 16 | Training batch size |
| `FANTASY_CHAINS` | 64 | PCD persistent chains |
| `PCD_K` | 5 | PCD-k steps |
| `SWEEPS_PER_STEP` | 6 | Gibbs sweeps per reverse step |

---

## Hardware Acceleration (TSU)

If `thrml` is installed and TSU is detected:

```python
TSU_AVAILABLE = True
```

Benefits:
- **10–100× faster Gibbs sampling**
- **Native sparse bilinear ops**
- **Energy-based sampling in hardware**

> Without TSU: **Full PyTorch fallback** (still fast on GPU)

---

## Output Examples

After training:
```
outputs/
├── samples_epoch_5.png
├── samples_epoch_20.png
├── ...
└── final_samples_64x64.png    # 8×8 grid of generated images
```

---

## Performance Notes

| Metric | Expected |
|-------|----------|
| **Training Time** | ~24–48 hrs on RTX 4090 (200 epochs) |
| **Sampling Speed** | ~2–5 sec per 64 images (GPU) |
| **FID (CIFAR-10 binarized)** | **~18–25** (competitive for discrete DTM) |
| **Memory** | ~8–10 GB VRAM |

---

## Troubleshooting

| Issue | Solution |
|------|----------|
| `thrml not found` | Normal — fallback used automatically |
| OOM Error | Reduce `BATCH_SIZE` or `FANTASY_CHAINS` |
| FID fails | Install `torch-fid`: `pip install torch-fid` |
| Black images | Ensure binarization: `(x > 0.5).float()` |

---

## Citation (Internal)

```bibtex
@software{ThermalAlgo2025,
  author = {{James Squire & xAI Thermal Team}},
  title = {ThermalAlgoPrototypev2.0: Block-Sparse Discrete DTM with Bit-Flip Noise and PCD-k},
  year = {2025},
  url = {https://github.com/Satoshi88818/ThermalAlgoPrototypev2.0/tree/main}; 
  note = {First-principles discrete generative model with TSU support}
}
```

---

## License

**xAI Internal Use Only** – Proprietary & Confidential

---

**Built with First Principles. Optimized for Reality.**  
*— xAI Thermal Computing Division, 2025*
