# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate MAPPO robustness with and without diffusion action denoising under Gaussian action noise.

Pipeline per step:
    1. MAPPO policy → clean actions
    2. Add Gaussian noise (adversarial perturbation)
    3. [Optional] Diffusion reverse pass to denoise actions
    4. env.step(actions)

Sweeps over --noise_stds × --t_start_list, outputs a comparison table.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Diffusion denoiser robustness sweep for MAPPO.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_episodes", type=int, default=200, help="Episodes per (noise_std, t_start) configuration.")
parser.add_argument("--task", type=str, default="Isaac-Cart-Double-Pendulum-Direct-v0")
parser.add_argument("--algorithm", type=str, default="MAPPO")
parser.add_argument("--agent", type=str, default=None)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--diffusion_model", type=str, required=True, help="Path to trained denoiser .pt checkpoint.")
parser.add_argument(
    "--noise_stds", type=float, nargs="+", default=[0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0],
    help="Gaussian noise std values applied to policy actions.",
)
parser.add_argument(
    "--noise_means", type=float, nargs="+", default=[0.0],
    help="List of Gaussian noise mean (bias) values to sweep over.",
)
parser.add_argument(
    "--t_start_list", type=int, nargs="+", default=[20, 40, 60],
    help="Reverse-diffusion start timesteps to evaluate. Larger = more denoising.",
)
parser.add_argument(
    "--attack_kind", type=str, default="gaussian", choices=["gaussian", "stuck_at", "delay"],
    help=(
        "Eval-time action attack to sweep: additive Gaussian noise, per-dim stuck-at actuator "
        "fault, or k-step action-delay fault."
    ),
)
parser.add_argument(
    "--sp_probs", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
    help="Per-dimension stuck probabilities to sweep over when --attack_kind stuck_at.",
)
parser.add_argument(
    "--delay_ks", type=int, nargs="+", default=[0, 1, 2, 3, 5, 8],
    help="Action-delay step counts to sweep over when --attack_kind delay.",
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, default="/tmp/diffusion_sweep_results.txt")
parser.add_argument("--success_threshold", type=float, default=None, help="Goal distance (m) for success metric.")
parser.add_argument("--reward_scale", type=float, default=2.0)
parser.add_argument("--dist_reward_scale", type=float, default=20.0)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import skrl
import torch
import torch.nn as nn
from packaging import version

SKRL_VERSION = "2.0.0"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(f"Requires skrl>={SKRL_VERSION}, got {skrl.__version__}")
    exit()

from skrl.utils.runner.torch import Runner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401

algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = args_cli.agent if args_cli.agent else f"skrl_{algorithm}_cfg_entry_point"


class StuckAtState:
    """Per-env, per-agent 'stuck-at' actuator fault, vectorized over num_envs.

    Each action dimension independently freezes (w.p. sp_prob) at the first
    action value computed after that env's episode starts. Frozen dims replay
    that value every step until the env's episode ends; non-frozen dims pass
    the policy's current action through unchanged. A fresh mask/frozen-value
    is (re-)sampled per env whenever that env starts a new episode.
    """

    def __init__(self, sp_prob: float, num_envs: int):
        self.sp_prob = sp_prob
        self.needs_init = np.ones(num_envs, dtype=bool)  # per-env, shared across agents
        self._mask = {}  # agent -> (num_envs, dim) bool
        self._val = {}  # agent -> (num_envs, dim)

    def apply(self, agent: str, action: torch.Tensor) -> torch.Tensor:
        if agent not in self._mask:
            self._mask[agent] = torch.zeros_like(action, dtype=torch.bool)
            self._val[agent] = torch.zeros_like(action)
        mask, val = self._mask[agent], self._val[agent]

        init_idx_np = np.nonzero(self.needs_init)[0]
        if init_idx_np.size:
            init_idx = torch.as_tensor(init_idx_np, device=action.device, dtype=torch.long)
            mask[init_idx] = torch.rand((init_idx.numel(), action.shape[-1]), device=action.device) < self.sp_prob
            val[init_idx] = action[init_idx]

        return torch.where(mask, val, action)

    def after_step(self):
        """Call once per step, after apply() has run for every agent."""
        self.needs_init[:] = False

    def on_episode_end(self, done_np: np.ndarray):
        """Call once per step, right after done_np is computed."""
        self.needs_init |= done_np


class DelayState:
    """Per-env, per-agent action-delay fault, vectorized over num_envs.

    Each executed action is the policy's action from delay_k steps ago (an
    actuation/communication delay), padded with the episode's first action
    until delay_k steps have elapsed. Implemented via a fixed-size per-agent
    ring buffer of length delay_k+1 (only that many past actions are ever
    needed).
    """

    def __init__(self, delay_k: int, num_envs: int):
        self.delay_k = delay_k
        self.n_slots = delay_k + 1
        self.t = np.zeros(num_envs, dtype=np.int64)  # steps since this env's episode reset
        self._buf = {}  # agent -> (num_envs, n_slots, dim)

    def apply(self, agent: str, action: torch.Tensor) -> torch.Tensor:
        if agent not in self._buf:
            self._buf[agent] = torch.zeros(
                action.shape[0], self.n_slots, action.shape[-1], device=action.device, dtype=action.dtype
            )
        buf = self._buf[agent]
        env_idx = torch.arange(action.shape[0], device=action.device)
        t = torch.as_tensor(self.t, device=action.device, dtype=torch.long)

        buf[env_idx, t % self.n_slots] = action  # append current action

        read_slot = torch.where(t >= self.delay_k, (t - self.delay_k) % self.n_slots, torch.zeros_like(t))
        return buf[env_idx, read_slot]

    def after_step(self):
        """Call once per step, after apply() has run for every agent."""
        self.t += 1

    def on_episode_end(self, done_np: np.ndarray):
        """Call once per step, right after done_np is computed."""
        self.t[done_np] = 0


# ---------------------------------------------------------------------------
# Denoiser (mirrors train_diffusion.py — keep in sync)
# ---------------------------------------------------------------------------

class ActionDenoiser(nn.Module):
    def __init__(self, action_dim: int, state_dim: int, hidden_dim: int = 256, T: int = 100):
        super().__init__()
        self.T = T
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(action_dim + hidden_dim * 2, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x_noisy, t, state):
        t_emb = self.time_mlp(t.float().unsqueeze(-1) / self.T)
        s_emb = self.state_mlp(state)
        h = torch.cat([x_noisy, t_emb, s_emb], dim=-1)
        return self.net(h)


def _make_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2):
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_bar


def load_denoiser(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    model = ActionDenoiser(
        action_dim=ckpt["action_dim"],
        state_dim=ckpt["state_dim"],
        hidden_dim=ckpt["hidden_dim"],
        T=ckpt["diffusion_steps"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    T = ckpt["diffusion_steps"]
    betas, alphas, alphas_bar = _make_beta_schedule(T)

    consts = {
        "betas": betas.to(device),
        "alphas": alphas.to(device),
        "alphas_bar": alphas_bar.to(device),
        "act_mean": ckpt["act_mean"].to(device),
        "act_std": ckpt["act_std"].to(device),
        "state_mean": ckpt["state_mean"].to(device),
        "state_std": ckpt["state_std"].to(device),
        "T": T,
    }
    print(f"[Diffusion] Loaded denoiser: action_dim={ckpt['action_dim']}, state_dim={ckpt['state_dim']}, T={T}")
    return model, consts


@torch.no_grad()
def diffusion_denoise_batched(
    noisy_actions: torch.Tensor,   # [B, Da]
    states: torch.Tensor,          # [B, Ds]
    t_start: int,
    model: ActionDenoiser,
    consts: dict,
    device: torch.device,
) -> torch.Tensor:
    """Run batched reverse diffusion from t_start → 0 to denoise actions.

    The noisy_actions tensor is treated as x_{t_start} directly (we skip the
    forward noising step and use the real adversarial noise as the starting point).
    """
    alphas = consts["alphas"]
    alphas_bar = consts["alphas_bar"]

    # Normalize
    x = (noisy_actions - consts["act_mean"]) / consts["act_std"]   # [B, Da]
    s = (states - consts["state_mean"]) / consts["state_std"]      # [B, Ds]

    x = x.to(device)
    s = s.to(device)

    B = x.shape[0]

    for t in reversed(range(t_start + 1)):
        t_tensor = torch.full((B,), t, dtype=torch.long, device=device)
        eps_pred = model(x, t_tensor, s)

        alpha = alphas[t]
        alpha_bar = alphas_bar[t]

        # DDPM reverse step: x_{t-1} = (x_t - sqrt(1-ā)*ε_pred) / sqrt(ā) rescaled
        x0_hat = (x - torch.sqrt(1.0 - alpha_bar) * eps_pred) / torch.sqrt(alpha_bar)
        x0_hat = x0_hat.clamp(-5.0, 5.0)  # guard against divergence

        if t > 0:
            noise = torch.randn_like(x)
            x = torch.sqrt(alpha) * x0_hat + torch.sqrt(1.0 - alpha) * noise
        else:
            x = x0_hat

    # Unnormalize
    return x * consts["act_std"] + consts["act_mean"]


# ---------------------------------------------------------------------------
# Episode evaluation helper
# ---------------------------------------------------------------------------

def run_one_sweep(
    env,
    runner,
    agent_order: Optional[List[str]],
    per_agent_dims: Optional[List[int]],
    is_marl: bool,
    num_envs: int,
    target_episodes: int,
    attack_kind: str,
    noise_std: float,
    noise_mean: float,
    sp_prob: float,
    delay_k: int,
    denoiser_model: Optional[ActionDenoiser],
    denoiser_consts: Optional[dict],
    t_start: int,
    device: torch.device,
    track_success: bool,
    success_threshold: float,
    reward_scale: float,
    dist_reward_scale: float,
) -> Tuple[float, float, float]:
    """Run `target_episodes` and return (mean_reward, std_reward, metric_pct)."""

    obs, _ = env.reset()
    states = env.state() if is_marl else None

    if attack_kind == "gaussian":
        attack_active = noise_std > 0.0 or noise_mean != 0.0
        attack_state = None
    elif attack_kind == "stuck_at":
        attack_active = sp_prob > 0.0
        attack_state = StuckAtState(sp_prob, num_envs)
    else:  # delay
        attack_active = delay_k > 0
        attack_state = DelayState(delay_k, num_envs)

    ep_reward_buf = np.zeros(num_envs, dtype=np.float64)
    ep_length_buf = np.zeros(num_envs, dtype=np.int32)
    ep_min_dist_buf = np.full(num_envs, np.inf, dtype=np.float64)

    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    episode_successes: List[bool] = []

    while len(episode_rewards) < target_episodes:
        with torch.inference_mode():
            outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
            if is_marl:
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in agent_order}
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])

            # --- Apply action attack ---
            if attack_kind == "gaussian":
                if attack_active:
                    if is_marl:
                        actions = {k: v + torch.randn_like(v) * noise_std + noise_mean for k, v in actions.items()}
                    else:
                        actions = actions + torch.randn_like(actions) * noise_std + noise_mean
            else:  # stuck_at, delay
                if is_marl:
                    actions = {k: attack_state.apply(k, v) for k, v in actions.items()}
                else:
                    actions = attack_state.apply("_single", actions)
                attack_state.after_step()

            # --- Diffusion denoising ---
            if denoiser_model is not None and attack_active:
                if is_marl:
                    action_vec = torch.cat([actions[a] for a in agent_order], dim=-1)  # [B, Da]
                    # env.state() may return a dict — concatenate to flat tensor
                    state_vec = torch.cat(list(states.values()), dim=-1) if isinstance(states, dict) else states
                else:
                    action_vec = actions
                    state_vec = obs

                clean_vec = diffusion_denoise_batched(
                    action_vec, state_vec, t_start, denoiser_model, denoiser_consts, device
                )

                if is_marl:
                    # Split back per agent
                    split = {}
                    cursor = 0
                    for a, dim in zip(agent_order, per_agent_dims):
                        split[a] = clean_vec[:, cursor: cursor + dim]
                        cursor += dim
                    actions = split
                else:
                    actions = clean_vec

            obs, rewards, terminated, truncated, _ = env.step(actions)
            if is_marl:
                states = env.state()

        if isinstance(rewards, dict):
            n_agents = len(rewards)
            r_np = sum(v.squeeze(-1).cpu().numpy() for v in rewards.values()) / n_agents
        else:
            r_np = rewards.squeeze(-1).cpu().numpy()

        if isinstance(terminated, dict):
            done_np = np.logical_or(
                sum(v.squeeze(-1).cpu().numpy() for v in terminated.values()) > 0,
                sum(v.squeeze(-1).cpu().numpy() for v in truncated.values()) > 0,
            )
        else:
            done_np = np.logical_or(terminated.squeeze(-1).cpu().numpy(), truncated.squeeze(-1).cpu().numpy())

        if attack_state is not None:
            attack_state.on_episode_end(done_np)

        ep_reward_buf += r_np
        ep_length_buf += 1

        if track_success:
            safe_r = np.clip(r_np / reward_scale, 1e-8, 1.0)
            step_dist = -np.log(safe_r) / dist_reward_scale
            ep_min_dist_buf = np.minimum(ep_min_dist_buf, step_dist)

        for i in range(num_envs):
            if done_np[i] and len(episode_rewards) < target_episodes:
                episode_rewards.append(float(ep_reward_buf[i]))
                episode_lengths.append(int(ep_length_buf[i]))
                if track_success:
                    episode_successes.append(bool(ep_min_dist_buf[i] < success_threshold))
                ep_reward_buf[i] = 0.0
                ep_length_buf[i] = 0
                ep_min_dist_buf[i] = np.inf

    rewards_arr = np.array(episode_rewards)
    lengths_arr = np.array(episode_lengths)
    max_len = int(lengths_arr.max()) if len(lengths_arr) else 1

    if track_success:
        metric = np.mean(episode_successes) * 100.0
    else:
        metric = (lengths_arr >= max_len).mean() * 100.0

    return float(rewards_arr.mean()), float(rewards_arr.std()), metric


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    experiment_cfg["seed"] = args_cli.seed

    log_root_path = os.path.abspath(
        os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    )
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_torch", other_dirs=["checkpoints"]
        )
    print(f"[INFO] MAPPO checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)
    runner.agent.load(resume_path)
    runner.agent.enable_training_mode(False, apply_to_models=True)

    is_marl = hasattr(env, "possible_agents")
    agent_order = list(env.possible_agents) if is_marl else None
    num_envs = args_cli.num_envs

    # Warm-up: single reset + forward pass to learn device and per-agent action dims
    obs_tmp, _ = env.reset()
    states_tmp = env.state() if is_marl else None
    # Get device from the first observation tensor
    obs_sample = next(iter(obs_tmp.values())) if isinstance(obs_tmp, dict) else obs_tmp
    device = obs_sample.device

    per_agent_dims: Optional[List[int]] = None
    if is_marl:
        with torch.inference_mode():
            out_tmp = runner.agent.act(obs_tmp, states_tmp, timestep=0, timesteps=0)
            acts_tmp = {a: out_tmp[-1][a].get("mean_actions", out_tmp[0][a]) for a in agent_order}
        per_agent_dims = [acts_tmp[a].shape[-1] for a in agent_order]
        print(f"[INFO] Per-agent action dims: {dict(zip(agent_order, per_agent_dims))}")

    # Load diffusion denoiser onto the same device as the env
    denoiser_model, denoiser_consts = load_denoiser(args_cli.diffusion_model, device)

    track_success = args_cli.success_threshold is not None
    success_threshold = args_cli.success_threshold or 0.0
    reward_scale = args_cli.reward_scale
    dist_reward_scale = args_cli.dist_reward_scale
    noise_stds = args_cli.noise_stds
    noise_means = args_cli.noise_means
    sp_probs = args_cli.sp_probs
    delay_ks = args_cli.delay_ks
    t_start_list = args_cli.t_start_list
    target_episodes = args_cli.num_episodes
    attack_kind = args_cli.attack_kind

    # Single-scalar-param attacks (stuck_at, delay) share one sweep-point/row shape.
    SINGLE_PARAM_ATTACKS = {"stuck_at": "sp_prob", "delay": "delay_k"}

    # Results: list of rows keyed by (noise_mean, noise_std), or by the single attack param.
    results = []

    if attack_kind == "gaussian":
        sweep_points = [{"noise_mean": nm, "noise_std": ns} for nm in noise_means for ns in noise_stds]
    else:
        param_name = SINGLE_PARAM_ATTACKS[attack_kind]
        param_values = sp_probs if attack_kind == "stuck_at" else delay_ks
        sweep_points = [{param_name: v} for v in param_values]

    for point in sweep_points:
        noise_mean = point.get("noise_mean", 0.0)
        noise_std = point.get("noise_std", 0.0)
        sp_prob = point.get("sp_prob", 0.0)
        delay_k = point.get("delay_k", 0)
        row = dict(point)
        if attack_kind == "gaussian":
            point_label = f"noise_mean={noise_mean:.2f}, noise_std={noise_std:.2f}"
        else:
            param_name = SINGLE_PARAM_ATTACKS[attack_kind]
            point_label = f"{param_name}={point[param_name]}"
        print(f"\n[INFO] {point_label}")

        # Baseline: no denoiser
        print(f"  Evaluating without denoiser ...")
        mean_r, std_r, metric = run_one_sweep(
            env=env, runner=runner, agent_order=agent_order, per_agent_dims=per_agent_dims,
            is_marl=is_marl, num_envs=num_envs, target_episodes=target_episodes,
            attack_kind=attack_kind, noise_std=noise_std, noise_mean=noise_mean, sp_prob=sp_prob,
            delay_k=delay_k, denoiser_model=None, denoiser_consts=None, t_start=0, device=device,
            track_success=track_success, success_threshold=success_threshold,
            reward_scale=reward_scale, dist_reward_scale=dist_reward_scale,
        )
        row["no_denoise"] = (mean_r, std_r, metric)
        print(f"    no_denoise: reward={mean_r:.2f}±{std_r:.2f}  metric={metric:.1f}%")

        # With denoiser at each t_start
        for t_start in t_start_list:
            print(f"  Evaluating with denoiser t_start={t_start} ...")
            mean_r, std_r, metric = run_one_sweep(
                env=env, runner=runner, agent_order=agent_order, per_agent_dims=per_agent_dims,
                is_marl=is_marl, num_envs=num_envs, target_episodes=target_episodes,
                attack_kind=attack_kind, noise_std=noise_std, noise_mean=noise_mean, sp_prob=sp_prob,
                delay_k=delay_k, denoiser_model=denoiser_model, denoiser_consts=denoiser_consts,
                t_start=t_start, device=device, track_success=track_success,
                success_threshold=success_threshold, reward_scale=reward_scale,
                dist_reward_scale=dist_reward_scale,
            )
            row[f"t{t_start}"] = (mean_r, std_r, metric)
            print(f"    t_start={t_start}: reward={mean_r:.2f}±{std_r:.2f}  metric={metric:.1f}%")

        results.append(row)

    env.close()

    # --- Build report ---
    metric_label = "success_rate" if track_success else "full_ep_rate"
    t_start_cols = [f"t{t}" for t in t_start_list]

    col_w = 22
    if attack_kind == "gaussian":
        header_parts = [f"{'noise_mean':<11}", f"{'noise_std':<10}"]
    else:
        header_parts = [f"{SINGLE_PARAM_ATTACKS[attack_kind]:<10}"]
    header_parts += [f"{'no_denoise':<{col_w}}"]
    for t in t_start_list:
        header_parts += [f"{'denoiser_t'+str(t):<{col_w}}"]
    header_line = " | ".join(header_parts)
    sep_line = "-" * len(header_line)

    ATTACK_TITLES = {
        "gaussian": "Diffusion Denoiser Sweep",
        "stuck_at": "Diffusion Denoiser Sweep — Stuck-At Actuator Fault",
        "delay": "Diffusion Denoiser Sweep — Action-Delay Fault",
    }
    title = ATTACK_TITLES[attack_kind]
    lines = [
        "",
        "=" * len(header_line),
        f"  {title} — {args_cli.task}",
        f"  {target_episodes} episodes × {num_envs} envs per configuration",
        f"  Metric: {metric_label} ({'< '+str(success_threshold)+' m' if track_success else 'reaches max len'})",
        "=" * len(header_line),
        f"  Format: mean_reward ± std  [{metric_label}%]",
        sep_line,
        "  " + header_line,
        "  " + sep_line,
    ]

    for row in results:
        if attack_kind == "gaussian":
            parts = [f"{row['noise_mean']:<11.2f}", f"{row['noise_std']:<10.2f}"]
        else:
            parts = [f"{row[SINGLE_PARAM_ATTACKS[attack_kind]]:<10.2f}"]
        for col in ["no_denoise"] + t_start_cols:
            mean_r, std_r, metric = row[col]
            cell = f"{mean_r:>6.2f}±{std_r:<5.2f} [{metric:>5.1f}%]"
            parts.append(f"{cell:<{col_w}}")
        lines.append("  " + " | ".join(parts))

    lines += ["  " + sep_line, ""]

    report = "\n".join(lines)
    with open(args_cli.out, "w") as f:
        f.write(report + "\n")
    print(report)
    print(f"[INFO] Results saved to {args_cli.out}")


if __name__ == "__main__":
    main()
    simulation_app.close()
