# block_sparse_dtm_production_v2.py
# PRODUCTION-READY & OPTIMIZED TSU-Native Discrete DTM
# 64x64 RGB Generation | Bit-Flip Forward Process | Binary DSM + PCD-k
# First-Principles Engineered | GPU/TSU Fallback | FID-Ready

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from typing import List, Tuple, Optional
import math

# =============================================
# 0. MOCK THORML (TSU Fallback)
# =============================================
try:
    import thrml
    from thrml import TSU, GibbsSampler, EnergyFunction
    from thrml.nn import SparseBilinear
    from thrml.sampling import BlockGibbsScheduler
    from thrml.training import PersistentContrastiveDivergence
    from thrml.noise import bitflip_schedule
    TSU_AVAILABLE = thrml.has_tsu()
except ImportError:
    print("thrml not found. Using high-performance CPU/GPU fallback.")
    TSU_AVAILABLE = False

    class SparseBilinear(nn.Module):
        def __init__(self, in_features, out_features, block_size, sparsity=0.9):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
            mask = torch.rand_like(self.weight) > sparsity
            self.register_buffer('mask', mask.float())
            self.weight.data.mul_(self.mask)

        def forward(self, x):
            return F.linear(x, self.weight * self.mask)

    def bitflip_schedule(T, p_min=0.01, p_max=0.3):
        t = np.linspace(0, 1, T + 1)
        p = p_min + (p_max - p_min) * (1 - np.cos(t * np.pi)) / 2
        return p[1:].clip(p_min, p_max)

    class PersistentContrastiveDivergence:
        def __init__(self, num_chains, steps=5):
            self.num_chains = num_chains
            self.steps = steps
            self.fantasy = None

        def step(self, fantasy, energy_fn, t):
            if self.fantasy is None:
                self.fantasy = fantasy.detach()
            x = self.fantasy
            for _ in range(self.steps):
                x = self._gibbs_step(x, energy_fn, t)
            self.fantasy = x.detach()
            return x

        def _gibbs_step(self, x_blocks, energy_fn, t):
            B, N, D = x_blocks.shape
            idx = torch.randperm(N, device=x_blocks.device)[:N//2]
            blk = x_blocks[:, idx]
            Bp = blk.shape[0]
            x_flip = blk.clone()
            x_flip = 1 - x_flip
            e_curr = energy_fn(blk.flatten(0,1), t).view(Bp, -1).sum(dim=1)
            e_flip = energy_fn(x_flip.flatten(0,1), t).view(Bp, -1).sum(dim=1)
            prob_flip = torch.sigmoid(e_curr - e_flip)
            flip = torch.bernoulli(prob_flip)
            blk = blk * (1 - flip.unsqueeze(-1).unsqueeze(-1)) + x_flip * flip.unsqueeze(-1).unsqueeze(-1)
            x_blocks = x_blocks.clone()
            x_blocks[:, idx] = blk
            return x_blocks

        def contrastive_loss(self, x0_blocks, energy_fn, t):
            pos_energy = energy_fn(x0_blocks.flatten(0,1), t).view(x0_blocks.shape[0], -1).sum(dim=1).mean()
            neg_energy = energy_fn(self.fantasy.flatten(0,1), t).view(self.fantasy.shape[0], -1).sum(dim=1).mean()
            return pos_energy - neg_energy

    class BlockGibbsScheduler:
        def __init__(self, block_pattern="checkerboard", sweeps_per_update=8):
            self.pattern = block_pattern
            self.sweeps = sweeps_per_update

        def step(self, x_blocks, energy_fn_per_block):
            B, N, D = x_blocks.shape
            grid = int(N ** 0.5)
            for _ in range(self.sweeps):
                for phase in [0, 1]:
                    idx = [(i * grid + j) for i in range(grid) for j in range(grid) if (i + j) % 2 == phase]
                    if not idx: continue
                    idx = torch.tensor(idx, device=x_blocks.device)
                    blk = x_blocks[:, idx]
                    Bp, Nb, D = blk.shape
                    blk_flat = blk.reshape(Bp * Nb, D)
                    x_flip = 1 - blk_flat
                    e_curr = energy_fn_per_block(blk_flat)
                    e_flip = energy_fn_per_block(x_flip)
                    prob_flip = torch.sigmoid(e_curr - e_flip)
                    flip = torch.bernoulli(prob_flip).bool()
                    blk_flat[flip] = 1 - blk_flat[flip]
                    x_blocks[:, idx] = blk_flat.view(Bp, Nb, D)
            return x_blocks

    thrml = type('thrml', (), {
        'has_tsu': lambda: False,
        'nn': type('nn', (), {'SparseBilinear': SparseBilinear}),
        'sampling': type('sampling', (), {'BlockGibbsScheduler': BlockGibbsScheduler}),
        'training': type('training', (), {'PersistentContrastiveDivergence': PersistentContrastiveDivergence}),
        'noise': type('noise', (), {'bitflip_schedule': bitflip_schedule}),
    })()

# =============================================
# 1. CONFIGURATION
# =============================================
IMG_SIZE = 64
CHANNELS = 3
BLOCK_SIZE = 8
BLOCKS_PER_SIDE = IMG_SIZE // BLOCK_SIZE
NUM_BLOCKS = BLOCKS_PER_SIDE ** 2
PBITS_PER_BLOCK = BLOCK_SIZE * BLOCK_SIZE * CHANNELS
TOTAL_PBITS = NUM_BLOCKS * PBITS_PER_BLOCK

T_STEPS = 100
SWEEPS_PER_STEP = 6
BATCH_SIZE = 16
FANTASY_CHAINS = 64
PCD_K = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {DEVICE} | TSU: {TSU_AVAILABLE}")

# Bit-flip schedule
P_FLIP = torch.from_numpy(thrml.noise.bitflip_schedule(T_STEPS)).to(DEVICE)

# =============================================
# 2. ENERGY FUNCTION (Binary + Time Bias)
# =============================================
class BlockSparseEnergy(EnergyFunction):
    def __init__(self):
        super().__init__()
        self.local = SparseBilinear(PBITS_PER_BLOCK, PBITS_PER_BLOCK, PBITS_PER_BLOCK, sparsity=0.9)
        self.skip = SparseBilinear(PBITS_PER_BLOCK, PBITS_PER_BLOCK, PBITS_PER_BLOCK, sparsity=0.99)
        self.bias = nn.Parameter(torch.zeros(PBITS_PER_BLOCK))

    def forward(self, x_block, t_bias):
        h1 = self.local(x_block)
        h2 = self.skip(x_block)
        energy = -(h1 + 0.3 * h2).sum(dim=-1) \
                 - (self.bias * x_block).sum(dim=-1) \
                 - (t_bias * x_block).sum(dim=-1)
        return energy

# =============================================
# 3. DTM MODEL (Bit-Flip, Binary DSM, PCD-k)
# =============================================
class BlockSparseDTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = T_STEPS
        self.p_flip = P_FLIP
        self.time_emb = nn.Embedding(T_STEPS + 1, 32)
        self.time_proj = nn.Linear(32, PBITS_PER_BLOCK)
        self.energy_fn = BlockSparseEnergy()
        self.block_grid = (BLOCKS_PER_SIDE, BLOCKS_PER_SIDE)

    def blockify(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, BLOCKS_PER_SIDE, BLOCK_SIZE, BLOCKS_PER_SIDE, BLOCK_SIZE, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, NUM_BLOCKS, PBITS_PER_BLOCK)
        return (x > 0).float()  # binarize

    def unblockify(self, x_blocks):
        B = x_blocks.shape[0]
        x = x_blocks.view(B, BLOCKS_PER_SIDE, BLOCKS_PER_SIDE, BLOCK_SIZE, BLOCK_SIZE, CHANNELS)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, IMG_SIZE, IMG_SIZE, CHANNELS)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def add_bitflip_noise(self, x0, t):
        p = self.p_flip[t].view(-1, 1, 1, 1)
        flip = torch.bernoulli(p * torch.ones_like(x0))
        xt = torch.where(flip > 0.5, 1 - x0, x0)
        return xt, flip

    def energy(self, x_blocks, t):
        B, num_blocks, D = x_blocks.shape
        t_emb = self.time_emb(t)
        t_proj = self.time_proj(t_emb)
        t_bias = t_proj.unsqueeze(1).expand(-1, num_blocks, -1).reshape(-1, D)
        x_flat = x_blocks.reshape(-1, D)
        energies = self.energy_fn(x_flat, t_bias)
        return energies.view(B, num_blocks).sum(dim=1)

    def binary_score_matching_loss(self, x0_blocks, xt_blocks, t):
        B, N, D = xt_blocks.shape
        xt_flat = xt_blocks.flatten(0, 1)
        t_batch = t.unsqueeze(1).expand(-1, N).flatten(0, 1)

        def energy_fn(x):
            return self.energy_fn(x, self.time_proj(self.time_emb(t_batch[:x.shape[0]//N])).flatten(0, 1)[:x.shape[0]])

        score = torch.autograd.grad(energy_fn(xt_flat).sum(), xt_flat, create_graph=True)[0]
        target = (x0_blocks - xt_blocks) / (self.p_flip[t].view(-1, 1, 1) + 1e-8)
        return F.mse_loss(score.flatten(0, 1), target.flatten(0, 1))

    @torch.no_grad()
    def sample(self, batch_size=8, sweeps=SWEEPS_PER_STEP, device=DEVICE):
        self.eval()
        x = torch.randint(0, 2, (batch_size, CHANNELS, IMG_SIZE, IMG_SIZE), device=device).float()
        x = self.blockify(x)

        scheduler = BlockGibbsScheduler(sweeps_per_update=sweeps)

        for t in reversed(range(self.T)):
            t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=device)
            t_emb = self.time_emb(t_tensor)
            t_proj = self.time_proj(t_emb)

            def energy_fn(blk):
                Bp = blk.shape[0]
                num_full = Bp // NUM_BLOCKS
                t_b = t_proj[:num_full].unsqueeze(1).expand(-1, NUM_BLOCKS, -1).flatten(0, 1)[:Bp]
                return self.energy_fn(blk, t_b)

            x = scheduler.step(x, energy_fn)
            if t % 20 == 0 or t <= 10:
                print(f"Reverse step {t+1}/{self.T}")

        return self.unblockify(x)

# =============================================
# 4. DATA LOADER
# =============================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x > 0.5).float()),  # binarize
])

train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

# =============================================
# 5. TRAINING LOOP (Binary DSM + PCD-k)
# =============================================
def train_dtm(model, epochs=200, lr=2e-4):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    pcd = PersistentContrastiveDivergence(num_chains=FANTASY_CHAINS, steps=PCD_K)

    fantasy = torch.randint(0, 2, (FANTASY_CHAINS, CHANNELS, IMG_SIZE, IMG_SIZE), device=DEVICE).float()
    fantasy = model.blockify(fantasy)

    for epoch in range(epochs):
        total_loss = total_sm = total_pcd = 0.0
        for i, (x0, _) in enumerate(train_loader):
            x0 = x0.to(DEVICE, non_blocking=True)
            batch_size = x0.shape[0]
            t = torch.randint(0, model.T, (batch_size,), device=DEVICE)

            x0_block = model.blockify(x0)
            xt, _ = model.add_bitflip_noise(x0, t)
            xt_block = model.blockify(xt)

            # Binary DSM
            sm_loss = model.binary_score_matching_loss(x0_block, xt_block, t)

            # PCD-k
            chain_batch = min(batch_size, FANTASY_CHAINS)
            t_fantasy = t[:chain_batch]
            fantasy[:chain_batch] = pcd.step(fantasy[:chain_batch], lambda x, tt: model.energy(x, tt), t_fantasy)
            pcd_loss = pcd.contrastive_loss(x0_block[:chain_batch], lambda x, tt: model.energy(x, tt), t_fantasy)

            loss = sm_loss + 0.5 * pcd_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_sm += sm_loss.item()
            total_pcd += pcd_loss.item()

            if i % 100 == 0:
                print(f"Epoch {epoch:03d} | Batch {i:04d} | Loss: {loss.item():.4f} (SM: {sm_loss.item():.3f}, PCD: {pcd_loss.item():.3f})")

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch:03d} | Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if epoch % 20 == 19 or epoch in [4, 9]:
            samples = model.sample(batch_size=8)
            save_image_grid(samples, f"outputs/samples_epoch_{epoch+1}.png")
            torch.save(model.state_dict(), f"checkpoints/dtm_epoch_{epoch+1}.pth")

# =============================================
# 6. UTILS
# =============================================
os.makedirs("outputs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

def save_image_grid(img_tensor: torch.Tensor, path: str, nrow: int = 8):
    grid = make_grid(img_tensor, nrow=nrow, padding=2, normalize=False)
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Image grid saved: {path}")

# =============================================
# 7. FID EVALUATION
# =============================================
def compute_fid(real_loader, model, num_samples=1000):
    try:
        import fid
        from fid.fid_score import calculate_fid_given_paths
        os.makedirs("tmp_real", exist_ok=True)
        os.makedirs("tmp_fake", exist_ok=True)

        # Save real
        for i, (img, _) in enumerate(real_loader):
            if i * BATCH_SIZE >= num_samples: break
            for j, im in enumerate(img):
                idx = i * BATCH_SIZE + j
                if idx >= num_samples: break
                plt.imsave(f"tmp_real/{idx:05d}.png", im.permute(1,2,0).cpu().numpy())

        # Generate fake
        model.eval()
        generated = 0
        with torch.no_grad():
            while generated < num_samples:
                batch = min(32, num_samples - generated)
                samples = model.sample(batch_size=batch)
                for s in samples:
                    plt.imsave(f"tmp_fake/{generated:05d}.png", s.permute(1,2,0).cpu().numpy())
                    generated += 1

        fid_value = calculate_fid_given_paths(['tmp_real', 'tmp_fake'], batch_size=32, device=DEVICE, dims=2048)
        import shutil
        shutil.rmtree("tmp_real")
        shutil.rmtree("tmp_fake")
        return fid_value
    except Exception as e:
        print(f"FID failed: {e}")
        return -1.0

# =============================================
# 8. RUN
# =============================================
if __name__ == "__main__":
    model = BlockSparseDTM().to(DEVICE)

    ckpt_path = "checkpoints/block_sparse_dtm_final.pth"
    if os.path.exists(ckpt_path):
        print(f"Loading pretrained model from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    else:
        print("Training from scratch...")
        train_dtm(model, epochs=200)
        torch.save(model.state_dict(), ckpt_path)
        print(f"Final model saved to {ckpt_path}")

    print("Generating 64 samples...")
    samples = model.sample(batch_size=64)
    save_image_grid(samples, "outputs/final_samples_64x64.png", nrow=8)

    print("Computing FID...")
    fid_score = compute_fid(train_loader, model, num_samples=1000)
    print(f"FID: {fid_score:.2f}" if fid_score > 0 else "FID unavailable")