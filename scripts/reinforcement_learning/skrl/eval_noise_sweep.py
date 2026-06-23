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

    is_marl = hasattr(env, "possible_agents")

    # --- noise sweep ---
    sweep_results = []  # list of (noise_std, mean_reward, std_reward, metric_value)

    for noise_std in noise_stds:
        print(f"[INFO] Evaluating noise_std={noise_std:.2f} ...")

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
                    actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a])
                               for a in env.possible_agents}
                    if noise_std > 0.0:
                        actions = {k: v + torch.randn_like(v) * noise_std
                                   for k, v in actions.items()}
                else:
                    actions = outputs[-1].get("mean_actions", outputs[0])
                    if noise_std > 0.0:
                        actions = actions + torch.randn_like(actions) * noise_std

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

        sweep_results.append((noise_std, rewards_arr.mean(), rewards_arr.std(), metric))
        print(f"  noise_std={noise_std:.2f}: reward={rewards_arr.mean():.2f}±{rewards_arr.std():.2f}  "
              f"{'success' if track_success else 'full-ep'}_rate={metric:.1f}%")

    env.close()

    # --- report ---
    metric_label = "success_rate" if track_success else "full_ep_rate"
    header_label = f"{'success_rate (< '+str(success_threshold)+' m)' if track_success else 'full-ep rate (max len)'}"

    lines = [
        "",
        "=" * 68,
        f"  Noise Robustness Sweep — {args_cli.task}",
        f"  {target_episodes} episodes × {num_envs} envs per noise level",
        f"  Noise injected in policy output space (before env.step)",
        "=" * 68,
        f"  {'noise_std':<10} | {'reward mean ± std':<22} | {metric_label}",
        f"  {'-'*10}-+-{'-'*22}-+-{'-'*12}",
    ]
    for noise_std, mean_r, std_r, metric in sweep_results:
        lines.append(f"  {noise_std:<10.2f} | {mean_r:>8.2f} ± {std_r:<11.2f} | {metric:>6.1f}%")
    lines.append("=" * 68)
    lines.append(f"  ({header_label})")
    lines.append("")

    report = "\n".join(lines)
    with open(args_cli.out, "w") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
    simulation_app.close()
