# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained MAPPO checkpoint under varying levels of Gaussian action noise."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Noise robustness sweep for a skrl MAPPO checkpoint.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments.")
parser.add_argument("--num_episodes", type=int, default=200, help="Episodes per noise level.")
parser.add_argument("--task", type=str, default="Isaac-Cart-Double-Pendulum-Direct-v0")
parser.add_argument("--algorithm", type=str, default="MAPPO")
parser.add_argument("--agent", type=str, default=None, help="Agent config entry point key.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (.pt).")
parser.add_argument(
    "--noise_stds",
    type=float,
    nargs="+",
    default=[0.0, 0.5, 1.0, 1.5, 2.0],
    help="List of Gaussian noise std values to sweep over (applied to actions before env.step).",
)
parser.add_argument(
    "--noise_means",
    type=float,
    nargs="+",
    default=[0.0],
    help="List of Gaussian noise mean (bias) values to sweep over.",
)
parser.add_argument(
    "--attack_kind",
    type=str,
    default="gaussian",
    choices=["gaussian", "stuck_at", "delay"],
    help=(
        "Eval-time action attack to sweep: additive Gaussian noise, per-dim stuck-at actuator "
        "fault, or k-step action-delay fault."
    ),
)
parser.add_argument(
    "--sp_probs",
    type=float,
    nargs="+",
    default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
    help="Per-dimension stuck probabilities to sweep over when --attack_kind stuck_at.",
)
parser.add_argument(
    "--delay_ks",
    type=int,
    nargs="+",
    default=[0, 1, 2, 3, 5, 8],
    help="Action-delay step counts to sweep over when --attack_kind delay.",
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, default="/tmp/noise_sweep_results.txt", help="Output file path.")
# Success rate via reward inversion: reward = reward_scale * exp(-dist_reward_scale * dist)
# For tasks without a natural success flag (Shadow Hand Over).
# If not set, uses full-episode rate (fraction of episodes reaching max length) — for Cart tasks.
parser.add_argument("--success_threshold", type=float, default=None, help="Goal distance threshold (m).")
parser.add_argument("--reward_scale", type=float, default=2.0)
parser.add_argument("--dist_reward_scale", type=float, default=20.0)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import gymnasium as gym
import numpy as np
import skrl
import torch
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


def collect_episodes(
    env,
    runner,
    is_marl: bool,
    num_envs: int,
    target_episodes: int,
    track_success: bool,
    success_threshold: float,
    reward_scale: float,
    dist_reward_scale: float,
    perturb_fn,
    on_done_fn=None,
):
    """Roll out one sweep point until target_episodes complete.

    perturb_fn(actions) -> actions: transforms the policy's mean-action(s)
    (dict[str, Tensor] if is_marl else Tensor) into the attacked actions
    actually sent to env.step().

    on_done_fn(done_np), optional: called once per step, right after done_np
    is computed, for attack state that needs to know which envs just
    auto-reset (e.g. stuck-at re-init).
    """
    obs, _ = env.reset()
    states = env.state() if is_marl else None

    ep_reward_buf = np.zeros(num_envs, dtype=np.float64)
    ep_length_buf = np.zeros(num_envs, dtype=np.int32)
    ep_min_dist_buf = np.full(num_envs, np.inf, dtype=np.float64)

    episode_rewards = []
    episode_lengths = []
    episode_successes = []

    while len(episode_rewards) < target_episodes:
        with torch.inference_mode():
            outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
            if is_marl:
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])

            actions = perturb_fn(actions)

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
            done_np = np.logical_or(
                terminated.squeeze(-1).cpu().numpy(),
                truncated.squeeze(-1).cpu().numpy(),
            )

        if on_done_fn is not None:
            on_done_fn(done_np)

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

    return rewards_arr, lengths_arr, metric


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
    print(f"[INFO] Checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)
    runner.agent.load(resume_path)
    runner.agent.enable_training_mode(False, apply_to_models=True)

    num_envs = args_cli.num_envs
    target_episodes = args_cli.num_episodes
    track_success = args_cli.success_threshold is not None
    success_threshold = args_cli.success_threshold or 0.0
    reward_scale = args_cli.reward_scale
    dist_reward_scale = args_cli.dist_reward_scale
    noise_stds = args_cli.noise_stds
    noise_means = args_cli.noise_means

    is_marl = hasattr(env, "possible_agents")
    metric_label = "success_rate" if track_success else "full_ep_rate"
    header_label = f"{'success_rate (< '+str(success_threshold)+' m)' if track_success else 'full-ep rate (max len)'}"

    # Single-scalar-param attacks (stuck_at, delay) share one sweep-loop/report shape.
    SINGLE_PARAM_ATTACKS = {
        "stuck_at": {
            "param_name": "sp_prob",
            "param_values": args_cli.sp_probs,
            "make_state": lambda v: StuckAtState(v, num_envs),
            "title": "Stuck-At Actuator Fault Robustness Sweep",
            "description": "Each action dim independently frozen w.p. sp_prob at its first post-reset value",
        },
        "delay": {
            "param_name": "delay_k",
            "param_values": args_cli.delay_ks,
            "make_state": lambda v: DelayState(v, num_envs),
            "title": "Action-Delay Robustness Sweep",
            "description": "Each action executed with a delay_k-step actuation delay (padded with the episode's first action)",
        },
    }

    gaussian_results = []  # (noise_mean, noise_std, mean_reward, std_reward, metric_value)
    single_param_results = []  # (param_value, mean_reward, std_reward, metric_value)

    if args_cli.attack_kind == "gaussian":
        for noise_mean in noise_means:
            for noise_std in noise_stds:
                print(f"[INFO] Evaluating noise_mean={noise_mean:.2f}, noise_std={noise_std:.2f} ...")

                def perturb(actions, noise_std=noise_std, noise_mean=noise_mean):
                    if noise_std > 0.0 or noise_mean != 0.0:
                        if isinstance(actions, dict):
                            return {k: v + torch.randn_like(v) * noise_std + noise_mean for k, v in actions.items()}
                        return actions + torch.randn_like(actions) * noise_std + noise_mean
                    return actions

                rewards_arr, _, metric = collect_episodes(
                    env, runner, is_marl, num_envs, target_episodes, track_success,
                    success_threshold, reward_scale, dist_reward_scale, perturb_fn=perturb,
                )
                gaussian_results.append((noise_mean, noise_std, rewards_arr.mean(), rewards_arr.std(), metric))
                print(f"  noise_mean={noise_mean:.2f}, noise_std={noise_std:.2f}: "
                      f"reward={rewards_arr.mean():.2f}±{rewards_arr.std():.2f}  "
                      f"{'success' if track_success else 'full-ep'}_rate={metric:.1f}%")
    else:
        attack_cfg = SINGLE_PARAM_ATTACKS[args_cli.attack_kind]
        param_name = attack_cfg["param_name"]
        for val in attack_cfg["param_values"]:
            print(f"[INFO] Evaluating {param_name}={val} ...")
            state = attack_cfg["make_state"](val)

            def perturb(actions, state=state):
                if isinstance(actions, dict):
                    out = {k: state.apply(k, v) for k, v in actions.items()}
                else:
                    out = state.apply("_single", actions)
                state.after_step()
                return out

            def on_done(done_np, state=state):
                state.on_episode_end(done_np)

            rewards_arr, _, metric = collect_episodes(
                env, runner, is_marl, num_envs, target_episodes, track_success,
                success_threshold, reward_scale, dist_reward_scale,
                perturb_fn=perturb, on_done_fn=on_done,
            )
            single_param_results.append((val, rewards_arr.mean(), rewards_arr.std(), metric))
            print(f"  {param_name}={val}: reward={rewards_arr.mean():.2f}±{rewards_arr.std():.2f}  "
                  f"{'success' if track_success else 'full-ep'}_rate={metric:.1f}%")

    env.close()

    # --- report ---
    if args_cli.attack_kind == "gaussian":
        lines = [
            "",
            "=" * 80,
            f"  Noise Robustness Sweep — {args_cli.task}",
            f"  {target_episodes} episodes × {num_envs} envs per (noise_mean, noise_std) pair",
            f"  Noise injected in policy output space (before env.step)",
            "=" * 80,
            f"  {'noise_mean':<11} | {'noise_std':<10} | {'reward mean ± std':<22} | {metric_label}",
            f"  {'-'*11}-+-{'-'*10}-+-{'-'*22}-+-{'-'*12}",
        ]
        for noise_mean, noise_std, mean_r, std_r, metric in gaussian_results:
            lines.append(
                f"  {noise_mean:<11.2f} | {noise_std:<10.2f} | {mean_r:>8.2f} ± {std_r:<11.2f} | {metric:>6.1f}%"
            )
        lines += ["=" * 80, f"  ({header_label})", ""]
    else:
        attack_cfg = SINGLE_PARAM_ATTACKS[args_cli.attack_kind]
        param_name = attack_cfg["param_name"]
        lines = [
            "",
            "=" * 80,
            f"  {attack_cfg['title']} — {args_cli.task}",
            f"  {target_episodes} episodes × {num_envs} envs per {param_name} value",
            f"  {attack_cfg['description']}",
            "=" * 80,
            f"  {param_name:<10} | {'reward mean ± std':<22} | {metric_label}",
            f"  {'-'*10}-+-{'-'*22}-+-{'-'*12}",
        ]
        for val, mean_r, std_r, metric in single_param_results:
            lines.append(f"  {val:<10.2f} | {mean_r:>8.2f} ± {std_r:<11.2f} | {metric:>6.1f}%")
        lines += ["=" * 80, f"  ({header_label})", ""]

    report = "\n".join(lines)
    with open(args_cli.out, "w") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
    simulation_app.close()
