# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Roll out a trained MAPPO checkpoint and save clean (state, action) pairs for diffusion training."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect clean (state, action) trajectories from a MAPPO checkpoint.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments.")
parser.add_argument("--num_episodes", type=int, default=500, help="Episodes to collect across all envs.")
parser.add_argument("--task", type=str, default="Isaac-Cart-Double-Pendulum-Direct-v0")
parser.add_argument("--algorithm", type=str, default="MAPPO")
parser.add_argument("--agent", type=str, default=None, help="Agent config entry point key (overrides --algorithm).")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (.pt). Latest best used if omitted.")
parser.add_argument("--output", type=str, default=None,
                    help="Output .npz path. Defaults to results/diffusion_data_<task_slug>.npz")
parser.add_argument("--seed", type=int, default=0)
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

    is_marl = hasattr(env, "possible_agents")
    num_envs = args_cli.num_envs
    target_episodes = args_cli.num_episodes

    # Determine output path
    if args_cli.output:
        out_path = os.path.abspath(args_cli.output)
    else:
        task_slug = args_cli.task.lower().replace("-", "_")
        out_path = os.path.abspath(os.path.join("results", f"diffusion_data_{task_slug}.npz"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Determine agent ordering for consistent action concatenation
    agent_order = list(env.possible_agents) if is_marl else None

    # Buffers
    state_buf = []
    action_buf = []

    ep_count = 0
    obs, _ = env.reset()
    states = env.state() if is_marl else None

    print(f"[Diffusion] Collecting {target_episodes} episodes from MAPPO policy (no noise) ...")

    while ep_count < target_episodes:
        with torch.inference_mode():
            outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
            if is_marl:
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in agent_order}
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])

        # Build flat state and action tensors [num_envs, D]
        if is_marl:
            # env.state() may return a dict keyed by agent — concatenate to flat tensor
            if isinstance(states, dict):
                state_vec = torch.cat(list(states.values()), dim=-1)            # [num_envs, Ds]
            else:
                state_vec = states                                               # [num_envs, Ds]
            action_vec = torch.cat([actions[a] for a in agent_order], dim=-1)   # [num_envs, Da]
        else:
            state_vec = obs                                                      # [num_envs, Ds]
            action_vec = actions                                                 # [num_envs, Da]

        state_buf.append(state_vec.cpu().numpy().astype(np.float32))
        action_buf.append(action_vec.cpu().numpy().astype(np.float32))

        obs, rewards, terminated, truncated, _ = env.step(actions if not is_marl else actions)
        if is_marl:
            states = env.state()

        if isinstance(terminated, dict):
            done_np = np.logical_or(
                sum(v.squeeze(-1).cpu().numpy() for v in terminated.values()) > 0,
                sum(v.squeeze(-1).cpu().numpy() for v in truncated.values()) > 0,
            )
        else:
            done_np = np.logical_or(terminated.squeeze(-1).cpu().numpy(), truncated.squeeze(-1).cpu().numpy())

        ep_count += int(done_np.sum())
        if ep_count % 50 < int(done_np.sum()):
            print(f"[Diffusion] Episodes collected: {ep_count}")

    env.close()

    states_arr = np.concatenate(state_buf, axis=0)    # [N_steps, Ds]
    actions_arr = np.concatenate(action_buf, axis=0)  # [N_steps, Da]

    print(f"[Diffusion] Dataset: states {states_arr.shape}, actions {actions_arr.shape}")
    np.savez(out_path, states=states_arr, actions=actions_arr)
    print(f"[Diffusion] Saved to {out_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
