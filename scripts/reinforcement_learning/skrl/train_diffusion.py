# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Train a DDPM step-level action denoiser from collected (state, action) pairs.

No Isaac Sim dependency — runs as plain: python train_diffusion.py --data_path <npz> --output <pt>
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ActionDenoiser(nn.Module):
    """DDPM noise predictor for step-level action denoising.

    Predicts the noise ε added to a clean action x_0 to produce x_t,
    conditioned on the current global state.

    Args:
        action_dim: Dimension of the (joint) action vector Da.
        state_dim:  Dimension of the global state vector Ds.
        hidden_dim: Width of all hidden layers.
        T:          Total diffusion steps (used only for time normalisation).
    """

    def __init__(self, action_dim: int, state_dim: int, hidden_dim: int = 256, T: int = 100):
        super().__init__()
        self.T = T

        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(action_dim + hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_noisy: [B, Da] — noisy action at timestep t
            t:       [B]     — diffusion timestep (integer, 0..T-1)
            state:   [B, Ds] — global state conditioning

        Returns:
            eps_pred: [B, Da] — predicted noise
        """
        t_emb = self.time_mlp(t.float().unsqueeze(-1) / self.T)  # [B, H]
        s_emb = self.state_mlp(state)                             # [B, H]
        h = torch.cat([x_noisy, t_emb, s_emb], dim=-1)           # [B, Da+2H]
        return self.net(h)                                         # [B, Da]


# ---------------------------------------------------------------------------
# DDPM schedule helpers
# ---------------------------------------------------------------------------

def make_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2):
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_bar


def q_sample(x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor,
             alphas_bar: torch.Tensor) -> torch.Tensor:
    """Forward diffusion q(x_t | x_0) = sqrt(ā_t)*x_0 + sqrt(1-ā_t)*ε"""
    a_bar = alphas_bar[t].view(-1, 1).to(x0.device)
    return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * eps


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    data = np.load(args.data_path)
    states_np = data["states"]    # [N, Ds]
    actions_np = data["actions"]  # [N, Da]

    N, Ds = states_np.shape
    _, Da = actions_np.shape
    print(f"[Diffusion] Dataset: {N} samples, state_dim={Ds}, action_dim={Da}")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[Diffusion] Device: {device}")

    states_t = torch.from_numpy(states_np).float()
    actions_t = torch.from_numpy(actions_np).float()

    # Normalize actions (z-score); save stats for inference
    act_mean = actions_t.mean(dim=0, keepdim=True)   # [1, Da]
    act_std = actions_t.std(dim=0, keepdim=True) + 1e-6
    actions_t = (actions_t - act_mean) / act_std

    # Normalize states too (helps conditioning)
    state_mean = states_t.mean(dim=0, keepdim=True)
    state_std = states_t.std(dim=0, keepdim=True) + 1e-6
    states_t = (states_t - state_mean) / state_std

    model = ActionDenoiser(
        action_dim=Da,
        state_dim=Ds,
        hidden_dim=args.hidden_dim,
        T=args.diffusion_steps,
    ).to(device)

    betas, alphas, alphas_bar = make_beta_schedule(args.diffusion_steps)
    alphas_bar = alphas_bar.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    batch_size = args.batch_size
    num_batches = max(1, N // batch_size)
    T = args.diffusion_steps

    print(f"[Diffusion] Training for {args.epochs} epochs, {num_batches} batches/epoch ...")
    for epoch in range(args.epochs):
        perm = torch.randperm(N)
        states_t = states_t[perm]
        actions_t = actions_t[perm]

        epoch_loss = 0.0
        for b in range(num_batches):
            start = b * batch_size
            end = min(N, (b + 1) * batch_size)

            x0 = actions_t[start:end].to(device)     # [B, Da]
            cond = states_t[start:end].to(device)     # [B, Ds]
            B = x0.shape[0]

            t = torch.randint(0, T, (B,), device=device)
            eps = torch.randn_like(x0)
            x_t = q_sample(x0, t, eps, alphas_bar)

            eps_pred = model(x_t, t, cond)
            loss = F.mse_loss(eps_pred, eps)

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item() * B

        epoch_loss /= N
        if (epoch + 1) % max(1, args.epochs // 10) == 0 or epoch == 0:
            print(f"[Diffusion] Epoch {epoch+1}/{args.epochs} — loss {epoch_loss:.6f}")

    # Save model + normalization stats
    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "action_dim": Da,
            "state_dim": Ds,
            "hidden_dim": args.hidden_dim,
            "diffusion_steps": T,
            "act_mean": act_mean,
            "act_std": act_std,
            "state_mean": state_mean,
            "state_std": state_std,
        },
        out_path,
    )
    print(f"[Diffusion] Model saved to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DDPM step-level action denoiser.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to .npz from collect_diffusion_data.py")
    parser.add_argument("--output", type=str, required=True, help="Output .pt checkpoint path")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden layer width (use 512 for Shadow Hand)")
    parser.add_argument("--diffusion_steps", type=int, default=100, help="Number of DDPM steps T")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default=None, help="torch device (default: auto-detect cuda/cpu)")
    args = parser.parse_args()
    train(args)
