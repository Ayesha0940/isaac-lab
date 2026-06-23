# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained MAPPO checkpoint over a fixed number of episodes and report metrics."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a skrl MAPPO checkpoint.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments.")
parser.add_argument("--num_episodes", type=int, default=200, help="Total episodes to evaluate across all envs.")
parser.add_argument("--task", type=str, default="Isaac-Cart-Double-Pendulum-Direct-v0")
parser.add_argument("--algorithm", type=str, default="MAPPO")
parser.add_argument("--agent", type=str, default=None, help="Agent config entry point key (overrides --algorithm default).")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (.pt). Latest best used if omitted.")
parser.add_argument("--video", action="store_true", default=False, help="Record a short video clip.")
parser.add_argument("--video_length", type=int, default=300, help="Video length in env steps.")
parser.add_argument("--video_width", type=int, default=1920, help="Video render width in pixels.")
parser.add_argument("--video_height", type=int, default=1080, help="Video render height in pixels.")
parser.add_argument("--camera_eye", type=float, nargs=3, default=None, help="Camera eye position (x y z).")
parser.add_argument("--camera_lookat", type=float, nargs=3, default=None, help="Camera lookat position (x y z).")
parser.add_argument("--video_dir", type=str, default=None, help="Override video output directory.")
parser.add_argument("--no_motion_blur", action="store_true", default=False, help="Disable motion blur in renderer.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, default="/tmp/eval_results.txt", help="File to write results to.")
# Success rate: fraction of episodes where object came within --success_threshold metres of goal.
# Inferred from per-step instantaneous reward using the env's reward formula:
#   reward = reward_scale * exp(-dist_reward_scale * goal_dist)
# Pass --reward_scale and --dist_reward_scale to match the task's reward config.
parser.add_argument("--success_threshold", type=float, default=None,
                    help="Distance threshold (m) for success. If set, success rate is reported.")
parser.add_argument("--reward_scale", type=float, default=2.0,
                    help="Coefficient in reward = reward_scale * exp(-dist_reward_scale * dist).")
parser.add_argument("--dist_reward_scale", type=float, default=20.0,
                    help="Exponent scale in reward formula (dist_reward_scale in env cfg).")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
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
# --agent overrides the default entry point derived from --algorithm
agent_cfg_entry_point = args_cli.agent if args_cli.agent else f"skrl_{algorithm}_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    experiment_cfg["seed"] = args_cli.seed

    if args_cli.video:
        env_cfg.viewer.resolution = (args_cli.video_width, args_cli.video_height)
        if args_cli.camera_eye:
            env_cfg.viewer.eye = tuple(args_cli.camera_eye)
        if args_cli.camera_lookat:
            env_cfg.viewer.lookat = tuple(args_cli.camera_lookat)

    log_root_path = os.path.abspath(
        os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    )
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_torch", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir
    print(f"[INFO] Checkpoint: {resume_path}")

    if args_cli.video and args_cli.no_motion_blur:
        import carb
        carb.settings.get_settings().set("/rtx/post/motionblur/maxBlurDiameterFraction", 0.0)
        carb.settings.get_settings().set("/rtx/post/motionblur/numSamples", 1)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_folder = args_cli.video_dir if args_cli.video_dir else os.path.join(log_dir, "videos", "eval")
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video to:", video_kwargs["video_folder"])
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)
    runner.agent.load(resume_path)
    runner.agent.enable_training_mode(False, apply_to_models=True)

    # --- evaluation loop ---
    num_envs = args_cli.num_envs
    target_episodes = args_cli.num_episodes
    track_success = args_cli.success_threshold is not None
    success_threshold = args_cli.success_threshold or 0.0
    reward_scale = args_cli.reward_scale
    dist_reward_scale = args_cli.dist_reward_scale

    episode_rewards = []      # cumulative reward per completed episode
    episode_lengths = []      # step count per completed episode
    episode_successes = []    # bool: did object reach within threshold of goal this episode?

    ep_reward_buf = np.zeros(num_envs, dtype=np.float64)
    ep_length_buf = np.zeros(num_envs, dtype=np.int32)
    # Per-env minimum goal distance seen this episode (inferred from instantaneous reward)
    ep_min_dist_buf = np.full(num_envs, np.inf, dtype=np.float64)

    obs, _ = env.reset()
    states = env.state()

    while len(episode_rewards) < target_episodes:
        with torch.inference_mode():
            outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            obs, rewards, terminated, truncated, _ = env.step(actions)
            states = env.state()

        # MARL wrapper returns dicts of shape (num_envs, 1) per agent — squeeze to (num_envs,)
        if isinstance(rewards, dict):
            # Both agents receive identical scalar rewards in cooperative tasks — take mean, not sum
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

        ep_reward_buf += r_np
        ep_length_buf += 1

        # Infer goal distance from instantaneous reward: dist = -ln(r / reward_scale) / dist_reward_scale
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

        if args_cli.video and len(episode_rewards) >= 1:
            break  # one episode is enough for the video

    env.close()

    # --- report ---
    rewards_arr = np.array(episode_rewards)
    lengths_arr = np.array(episode_lengths)
    max_len = int(lengths_arr.max()) if len(lengths_arr) else 1

    lines = [
        "",
        "=" * 60,
        f"  MAPPO Evaluation — {len(episode_rewards)} episodes, {num_envs} envs",
        "=" * 60,
        f"  Episode reward  : {rewards_arr.mean():.2f} ± {rewards_arr.std():.2f}",
        f"                    min={rewards_arr.min():.2f}  max={rewards_arr.max():.2f}",
        f"  Episode length  : {lengths_arr.mean():.1f} ± {lengths_arr.std():.1f}  (max possible: {max_len})",
    ]
    if track_success:
        sr = np.mean(episode_successes) * 100
        lines.append(f"  Success rate    : {sr:.1f}%  (min goal dist < {success_threshold:.2f} m at any step)")
    else:
        sr_by_length = (lengths_arr >= max_len).mean() * 100
        lines.append(f"  Full-ep rate    : {sr_by_length:.1f}%  (episodes reaching max length)")
    lines.append("=" * 60)

    report = "\n".join(lines)
    out_path = args_cli.out
    with open(out_path, "w") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
    simulation_app.close()
