
import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader, Dataset


# -------------------------
# Repro utils
# -------------------------

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Scaling (stable for GAN)
# -------------------------
@dataclass
class MinMaxScalerNP:
    min_: np.ndarray = None
    max_: np.ndarray = None
    scale_: np.ndarray = None
    constant_mask_: np.ndarray = None
    eps: float = 1e-9

    def fit(self, x: np.ndarray):
        self.min_ = np.nanmin(x, axis=0)
        self.max_ = np.nanmax(x, axis=0)

        rng = self.max_ - self.min_
        self.constant_mask_ = rng < self.eps
        self.scale_ = np.where(self.constant_mask_, 1.0, rng)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x01 = (x - self.min_) / self.scale_
        y = (x01 * 2.0) - 1.0
        y[:, self.constant_mask_] = 0.0
        return y

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        x01 = (y + 1.0) / 2.0
        x = x01 * self.scale_ + self.min_
        x[:, self.constant_mask_] = self.min_[self.constant_mask_]
        return x

# -------------------------
# Dataset
# -------------------------

class PairDataset(Dataset):
    def __init__(self, x):
        x = torch.from_numpy(x.astype(np.float32))
        self.z1 = x
        self.z2 = x[torch.randperm(len(x))]  # 打乱作为另一分布

    def __len__(self):
        return self.z1.shape[0]

    def __getitem__(self, idx):
        return self.z1[idx], self.z2[idx]


# -------------------------
# Self-Attention (SAGAN style) over "feature tokens"
# -------------------------

class SelfAttention1D(nn.Module):
    """
    Self-attention over sequence length L with channels C.
    Input: (B, C, L)
    """
    def __init__(self, in_channels: int):
        super().__init__()
        c_q = max(1, in_channels // 8)
        self.query = spectral_norm(nn.Conv1d(in_channels, c_q, kernel_size=1))
        self.key   = spectral_norm(nn.Conv1d(in_channels, c_q, kernel_size=1))
        self.value = spectral_norm(nn.Conv1d(in_channels, in_channels, kernel_size=1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (B, C, L)
        B, C, L = x.shape
        q = self.query(x).permute(0, 2, 1)      # (B, L, Cq)
        k = self.key(x)                          # (B, Cq, L)
        attn = torch.bmm(q, k)                   # (B, L, L)
        attn = F.softmax(attn / math.sqrt(k.shape[1]), dim=-1)
        v = self.value(x)                        # (B, C, L)
        out = torch.bmm(v, attn.permute(0, 2, 1))  # (B, C, L)
        return x + self.gamma * out


# -------------------------
# Generator / Discriminator
# -------------------------

class Generator(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 256, seq_channels: int = 64):
        super().__init__()
        self.feat_dim = feat_dim
        self.fc1 = nn.Linear(feat_dim, hidden)
        self.fc2 = nn.Linear(hidden, seq_channels * feat_dim)
        self.attn = SelfAttention1D(seq_channels)
        self.out = nn.Conv1d(seq_channels, 1, kernel_size=1)

    def forward(self, x):
        h = F.leaky_relu(self.fc1(x), 0.2)
        h = F.relu(self.fc2(h))
        h = h.view(x.shape[0], -1, self.feat_dim)
        h = self.attn(h)
        delta = self.out(h).squeeze(1)
        return torch.tanh(delta)


class Discriminator(nn.Module):
    """
    Map features -> (C, L), self-attn, then MLP head.
    Spectral norm on layers (SAGAN common practice).
    """
    def __init__(self, feat_dim: int, hidden: int = 256, seq_channels: int = 64):
        super().__init__()
        self.feat_dim = feat_dim
        self.seq_channels = seq_channels

        self.inp = spectral_norm(nn.Conv1d(1, seq_channels, kernel_size=1))
        self.attn = SelfAttention1D(seq_channels)

        self.conv = nn.Sequential(
            spectral_norm(nn.Conv1d(seq_channels, seq_channels, kernel_size=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv1d(seq_channels, seq_channels, kernel_size=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = spectral_norm(nn.Linear(seq_channels * feat_dim, hidden))
        self.out = spectral_norm(nn.Linear(hidden, 1))

    def forward(self, x):
        # x: (B, L) scaled to [-1, 1]
        h = x.unsqueeze(1)       # (B, 1, L)
        h = self.inp(h)          # (B, C, L)
        h = self.attn(h)
        h = self.conv(h)
        h = h.reshape(h.shape[0], -1)
        h = F.leaky_relu(self.fc(h), 0.2, inplace=True)
        return self.out(h)


# -------------------------
# Hinge GAN losses (SAGAN common)
# -------------------------

def d_hinge_loss(d_real, d_fake):
    return torch.mean(F.relu(1.0 - d_real)) + torch.mean(F.relu(1.0 + d_fake))

def g_hinge_loss(d_fake):
    return -torch.mean(d_fake)


# -------------------------
# IO helpers
# -------------------------

def save_run_config(out_dir: str, cfg: Dict):
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

def load_run_config(run_dir: str) -> Dict:
    with open(os.path.join(run_dir, "config.json"), "r", encoding="utf-8") as f:
        return json.load(f)

def save_scaler(out_dir: str, scaler: MinMaxScalerNP, cols: List[str]):
    np.save(os.path.join(out_dir, "scaler_min.npy"), scaler.min_)
    np.save(os.path.join(out_dir, "scaler_max.npy"), scaler.max_)
    with open(os.path.join(out_dir, "columns.json"), "w", encoding="utf-8") as f:
        json.dump(cols, f, indent=2)

def load_scaler(run_dir: str) -> Tuple[MinMaxScalerNP, List[str]]:
    s = MinMaxScalerNP()

    s.min_ = np.load(os.path.join(run_dir, "scaler_min.npy"))
    s.max_ = np.load(os.path.join(run_dir, "scaler_max.npy"))

    rng = s.max_ - s.min_
    s.constant_mask_ = rng < s.eps
    s.scale_ = np.where(s.constant_mask_, 1.0, rng)

    with open(os.path.join(run_dir, "columns.json"), "r", encoding="utf-8") as f:
        cols = json.load(f)

    return s, cols


# -------------------------
# Train
# -------------------------

def train_sagan(
    z_csv: str,
    out_dir: str,
    z_dim: int = 128,
    epochs: int = 200,
    batch_size: int = 512,
    lr_g: float = 2e-4,
    lr_d: float = 2e-4,
    n_critic: int = 1,
    hidden: int = 256,
    seq_channels: int = 64,
    seed: int = 42,
):
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(z_csv)
    df = df.select_dtypes(include=[np.number]).copy()
    cols = df.columns.tolist()

    X = df.to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = MinMaxScalerNP()
    scaler.fit(X)
    Xs = scaler.transform(X)

    print("X nan:", np.isnan(X).sum(), "X inf:", np.isinf(X).sum())
    print("Xs nan:", np.isnan(Xs).sum(), "Xs inf:", np.isinf(Xs).sum())
    print("Xs min/max:", np.nanmin(Xs), np.nanmax(Xs))

    bad_cols = []
    for i, c in enumerate(cols):
        if np.isnan(Xs[:, i]).any() or np.isinf(Xs[:, i]).any():
            bad_cols.append(c)

    print("bad cols:", bad_cols[:50], "count:", len(bad_cols))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat_dim = Xs.shape[1]
    G = Generator(feat_dim=feat_dim, hidden=hidden, seq_channels=seq_channels).to(device)
    D = Discriminator(feat_dim=feat_dim, hidden=hidden, seq_channels=seq_channels).to(device)

    # TTUR-ish (optionally different lrs), hinge uses betas (0, 0.9) commonly
    opt_g = torch.optim.RMSprop(G.parameters(), lr=3e-4, alpha=0.9)
    opt_d = torch.optim.RMSprop(D.parameters(), lr=3e-4, alpha=0.9)

    dl = DataLoader(PairDataset(Xs), batch_size=batch_size, shuffle=True, drop_last=True)

    cfg = dict(
        z_csv=z_csv,
        z_dim=z_dim,
        epochs=epochs,
        batch_size=batch_size,
        lr_g=lr_g,
        lr_d=lr_d,
        n_critic=n_critic,
        hidden=hidden,
        seq_channels=seq_channels,
        seed=seed,
        feat_dim=feat_dim,
        device=device,
    )
    save_run_config(out_dir, cfg)
    save_scaler(out_dir, scaler, cols)

    step = 0
    for ep in range(1, epochs + 1):
        for z1, z2 in dl:
            z1 = z1.to(device)
            z2 = z2.to(device)

            # ---- D ----
            fake = G(z1).detach()
            d_real = D(z2)
            d_fake = D(fake)
            loss_d = d_hinge_loss(d_real, d_fake)

            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()

            # ---- G ----
            fake = G(z1)
            d_fake = D(fake)
            loss_g = g_hinge_loss(d_fake)

            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

            step += 1

        if ep == 1 or ep % max(1, epochs // 10) == 0:
            print(f"Epoch {ep:4d}/{epochs} | D: {loss_d.item():.4f} | G: {loss_g.item():.4f}")
            torch.save({"G": G.state_dict(), "D": D.state_dict()}, os.path.join(out_dir, f"ckpt_ep{ep}.pt"))

    torch.save({"G": G.state_dict(), "D": D.state_dict()}, os.path.join(out_dir, "ckpt_last.pt"))
    print(f"Saved run to: {out_dir}")


# -------------------------
# Generate + Restore boundary buses
# -------------------------
@torch.no_grad()
def generate_equated(run_dir: str, x_csv: str, batch: int = 2048) -> pd.DataFrame:
    cfg = load_run_config(run_dir)
    scaler, cols = load_scaler(run_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat_dim = cfg["feat_dim"]

    G = Generator(
        feat_dim=feat_dim,
        hidden=cfg["hidden"],
        seq_channels=cfg["seq_channels"]
    ).to(device)

    ckpt = torch.load(os.path.join(run_dir, "ckpt_last.pt"), map_location=device)
    G.load_state_dict(ckpt["G"])
    G.eval()

    # --- load input data ---
    df = pd.read_csv(x_csv)
    df = df.select_dtypes(include=[np.number]).copy()

    X = df.to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # --- scale ---
    Xs = scaler.transform(X)

    # --- generate ---
    outs = []
    for i in range(0, len(Xs), batch):
        xb = Xs[i:i+batch]
        xb_tensor = torch.from_numpy(xb.astype(np.float32)).to(device)

        fake_scaled = G(xb_tensor).cpu().numpy()
        outs.append(fake_scaled)

    X_scaled = np.vstack(outs)

    # --- inverse ---
    X_out = scaler.inverse_transform(X_scaled)
    X_out = np.clip(X_out, scaler.min_, scaler.max_)

    return pd.DataFrame(X_out, columns=cols)

def restore_boundary_buses(
    z_equated: pd.DataFrame,
    tie_csv: str,
    mode: str = "sample",
    tie_row_index: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Paper: restore P/Q at boundary buses by adding tie-line aggregated flows back.

    Inputs:
      - z_equated: generated equated samples (contains columns like P100.., Q100.. for attack region)
      - tie_csv: Tie_Line_Flows_15k.csv (your notebook shifted +1 already; columns like P100/Q100 for boundary buses)
      - mode:
          "sample": for each generated row, randomly sample one tie-flow row from tie_csv and add it
          "fixed":  use one specific tie_row_index for all generated rows (simulate a single operating point)
    """
    rng = np.random.default_rng(seed)
    tie = pd.read_csv(tie_csv).select_dtypes(include=[np.number]).copy()
    tie_cols = tie.columns.tolist()

    # only restore columns that exist in both
    common = [c for c in tie_cols if c in z_equated.columns]
    if not common:
        raise ValueError("No overlapping P/Q columns between tie_csv and z_equated. "
                         "Check your tie_csv shifting (+1) and your z columns.")

    if mode == "fixed":
        if tie_row_index is None:
            raise ValueError("mode='fixed' requires --tie_row_index")
        tie_rows = tie.loc[[tie_row_index], common].to_numpy(dtype=np.float32)
        add_mat = np.repeat(tie_rows, repeats=len(z_equated), axis=0)
    elif mode == "sample":
        idx = rng.integers(low=0, high=len(tie), size=len(z_equated))
        add_mat = tie.iloc[idx][common].to_numpy(dtype=np.float32)
    else:
        raise ValueError("mode must be 'sample' or 'fixed'")

    z_restored = z_equated.copy()
    z_restored.loc[:, common] = z_restored.loc[:, common].to_numpy(dtype=np.float32) + add_mat
    return z_restored


# -------------------------
# CLI
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--z_csv", type=str, required=True)
    p_train.add_argument("--out_dir", type=str, required=True)
    p_train.add_argument("--z_dim", type=int, default=128)
    p_train.add_argument("--epochs", type=int, default=200)
    p_train.add_argument("--batch_size", type=int, default=512)
    p_train.add_argument("--lr_g", type=float, default=2e-4)
    p_train.add_argument("--lr_d", type=float, default=2e-4)
    p_train.add_argument("--n_critic", type=int, default=1)
    p_train.add_argument("--hidden", type=int, default=256)
    p_train.add_argument("--seq_channels", type=int, default=64)
    p_train.add_argument("--seed", type=int, default=42)

    p_gen = sub.add_parser("gen")
    p_gen.add_argument("--run_dir", type=str, required=True)
    p_gen.add_argument("--x_csv", type=str, required=True)
    p_gen.add_argument("--n", type=int, required=True)
    p_gen.add_argument("--out_equated_csv", type=str, required=True)
    p_gen.add_argument("--tie_csv", type=str, required=True)
    p_gen.add_argument("--out_restored_csv", type=str, required=True)
    p_gen.add_argument("--restore_mode", type=str, choices=["sample", "fixed"], default="sample")
    p_gen.add_argument("--tie_row_index", type=int, default=None)
    p_gen.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.cmd == "train":
        train_sagan(
            z_csv=args.z_csv,
            out_dir=args.out_dir,
            z_dim=args.z_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr_g=args.lr_g,
            lr_d=args.lr_d,
            n_critic=args.n_critic,
            hidden=args.hidden,
            seq_channels=args.seq_channels,
            seed=args.seed,
        )
    elif args.cmd == "gen":
        # 1. 生成 equated attack data（基于输入 csv）
        z_eq = generate_equated(args.run_dir, args.x_csv)
        z_eq.to_csv(args.out_equated_csv, index=False)

        # 2. restore boundary bus
        z_restored = restore_boundary_buses(
            z_equated=z_eq,
            tie_csv=args.tie_csv,
            mode=args.restore_mode,
            tie_row_index=args.tie_row_index,
            seed=args.seed,
        )
        z_restored.to_csv(args.out_restored_csv, index=False)

        print(f"Saved equated attack to: {args.out_equated_csv}")
        print(f"Saved final attack to:   {args.out_restored_csv}")
        print(f"Saved equated z_attack to:  {args.out_equated_csv}")
        print(f"Saved restored z_attack to: {args.out_restored_csv}")


if __name__ == "__main__":
    main()