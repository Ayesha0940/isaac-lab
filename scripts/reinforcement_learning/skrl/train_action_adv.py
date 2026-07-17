# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Train the skrl MAPPO baseline with action-adversarial training (RARL-style).

A learned adversary (a single skrl PPO agent, observing the environment's centralized
state) outputs a bounded perturbation added to the protagonist agents' actions before
each environment step. The adversary is trained to *minimize* the protagonist's reward
(zero-sum), while the protagonist is trained normally (MAPPO/PPO) on the reward it
receives from the perturbed step. This exposes the protagonist to adversarial action
noise during training, rather than only at evaluation time (see eval_noise_sweep.py),
so the resulting policy is inherently more robust to action perturbations.

Semantic note (on-policy vs. off-policy adversarial training):
This is a port of an MADDPG-based (off-policy) action-adversarial recipe to an
on-policy MAPPO/PPO setting. MADDPG's replay buffer can store the actually-applied
(perturbed) action because its critic just regresses TD targets. MAPPO/PPO cannot: its
`act()` caches the log-probability of the action it sampled, and its surrogate
objective needs `actions` recorded in memory to match that cached log-probability, or
the importance-sampling ratio is invalid. So the protagonist's transition is recorded
with its own CLEAN (unperturbed) action, while the environment is stepped with the
PERTURBED action -- the standard Pinto et al. 2017 "Robust Adversarial RL" formulation.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train MAPPO with action-adversarial training (skrl).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to protagonist checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--algorithm",
    type=str,
    default="MAPPO",
    choices=["MAPPO", "mappo"],
    help="Algorithm to use for the protagonist (adversarial training currently supports MAPPO only).",
)

# --- adversary args ---
parser.add_argument(
    "--constraint-epsilon", type=float, default=0.1, help="Bound of the adversary's action perturbation."
)
parser.add_argument("--adv-lr", type=float, default=3.0e-4, help="Adversary PPO learning rate.")
parser.add_argument("--adv-rollouts", type=int, default=16, help="Adversary PPO rollout length.")
parser.add_argument("--adv-learning-epochs", type=int, default=8, help="Adversary PPO learning epochs per update.")
parser.add_argument("--adv-mini-batches", type=int, default=1, help="Adversary PPO mini-batches per update.")
parser.add_argument(
    "--adv-hidden-units", type=int, nargs="+", default=[64, 64], help="Adversary policy/value MLP layer sizes."
)
parser.add_argument("--adv-entropy-scale", type=float, default=0.0, help="Adversary PPO entropy loss scale.")
parser.add_argument("--adv-checkpoint", type=str, default=None, help="Path to adversary checkpoint to resume training.")
parser.add_argument(
    "--adv-warmup-iterations",
    type=int,
    default=0,
    help="Number of initial protagonist rollouts during which the adversary's perturbation is forced to zero.",
)
parser.add_argument(
    "--reward-clip",
    type=float,
    default=5.0,
    help=(
        "Clip per-step rewards to [-reward_clip, reward_clip] before they reach either agent's "
        "record_transition(). A rare large reward outlier can collapse PPO's advantage-normalization "
        "std and explode the policy loss in a single update; this guards both the protagonist's and "
        "the adversary's (negated) reward stream against that. Set to 0 to disable."
    ),
)
parser.add_argument(
    "--adv-kl-threshold",
    type=float,
    default=0.008,
    help=(
        "KL-adaptive learning-rate threshold for the adversary (mirrors the scheduler already used by "
        "the protagonist's own yaml config). The adversary chases a non-stationary target (the "
        "protagonist keeps improving throughout training), which can make its value/policy loss ramp up "
        "and eventually explode over hundreds of updates; an adaptive LR shrinks step size automatically "
        "once updates start moving the policy too far, the same safeguard that already keeps the "
        "protagonist stable on this task. Set to 0 to disable (fixed --adv-lr throughout)."
    ),
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
args_cli.algorithm = args_cli.algorithm.upper()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import random
import sys
import time
from datetime import datetime

import gymnasium as gym
import gymnasium.spaces as gym_spaces
import skrl
import torch
import tqdm
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "2.0.0"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainerCfg
from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model
from skrl.utils.runner.torch import Runner

from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

algorithm = args_cli.algorithm.lower()
if algorithm != "mappo":
    raise ValueError(f"Action-adversarial training only supports MAPPO, got --algorithm {args_cli.algorithm}")
agent_cfg_entry_point = args_cli.agent if args_cli.agent else f"skrl_{algorithm}_cfg_entry_point"


def _first_non_finite_param(model: torch.nn.Module) -> str | None:
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            return name
    return None


def _assert_agent_finite(agent, label: str, t: int) -> None:
    """Fail fast if an agent's policy/value weights have gone NaN/Inf.

    PPO updates that diverge (e.g. a reward outlier collapsing the advantage-
    normalization std) corrupt every subsequent checkpoint silently -- training
    otherwise keeps running for the full duration on a dead network. Checking
    right after each agent's own update cadence catches this within one rollout
    instead of burning the rest of the run.

    Handles both a single-agent PPO (`.policy`/`.value`) and a MultiAgent like
    MAPPO (`.policies`/`.values` dicts keyed by uid).
    """
    if hasattr(agent, "policies"):
        models = list(agent.policies.values()) + list(agent.values.values())
    else:
        models = [agent.policy, agent.value]

    bad_param = None
    for model in models:
        if model is None:
            continue
        bad_param = _first_non_finite_param(model)
        if bad_param:
            break
    if bad_param:
        raise RuntimeError(
            f"[NaN-guard] {label} weights became non-finite at timestep {t} (first bad param: {bad_param}). "
            "Aborting training early -- consider a smaller --constraint-epsilon, a larger "
            "--adv-warmup-iterations, or a lower --adv-lr/learning_rate."
        )


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: DirectMARLEnvCfg, agent_cfg: dict):
    """Train MAPPO adversarially with skrl."""
    if not isinstance(env_cfg, DirectMARLEnvCfg):
        raise ValueError("Action-adversarial training requires a multi-agent (DirectMARLEnvCfg) task.")

    # override configurations with non-hydra CLI arguments (mirrors train.py)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # --- logging directory setup (identical convention to train.py, so checkpoints stay
    # discoverable by the existing eval_mappo.py / eval_noise_sweep.py get_checkpoint_path scan) ---
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_torch"
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f"_{agent_cfg['agent']['experiment']['experiment_name']}"
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    protagonist_resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    # --- build the protagonist (stock skrl Runner, exactly like train.py) ---
    # We never call runner.run(): we only want Runner to build/initialize `runner.agent`
    # (memories, preprocessors, checkpoint/log directories) exactly as a stock MAPPO run
    # would, then we drive the interaction loop by hand so the adversary can be spliced in.
    runner = Runner(env, agent_cfg)
    if protagonist_resume_path:
        print(f"[INFO] Loading protagonist checkpoint from: {protagonist_resume_path}")
        runner.agent.load(protagonist_resume_path)
    protagonist = runner.agent

    # --- build the adversary: a single skrl PPO agent perturbing the joint action ---
    possible_agents = env.possible_agents
    total_action_dim = sum(int(env.action_spaces[uid].shape[0]) for uid in possible_agents)
    state_dim = int(next(iter(env.state_spaces.values())).shape[0])
    device = env.device

    adv_observation_space = gym_spaces.Box(low=-float("inf"), high=float("inf"), shape=(state_dim,))
    adv_action_space = gym_spaces.Box(
        low=-args_cli.constraint_epsilon, high=args_cli.constraint_epsilon, shape=(total_action_dim,)
    )

    adv_network = [{"name": "net", "input": "STATES", "layers": args_cli.adv_hidden_units, "activations": "elu"}]
    adv_policy = gaussian_model(
        observation_space=adv_observation_space,
        state_space=adv_observation_space,
        action_space=adv_action_space,
        device=device,
        clip_actions=True,
        clip_log_std=True,
        min_log_std=-20.0,
        max_log_std=2.0,
        initial_log_std=0.0,
        network=adv_network,
        output="ACTIONS",
    )
    adv_value = deterministic_model(
        observation_space=adv_observation_space,
        state_space=adv_observation_space,
        device=device,
        network=adv_network,
        output="ONE",
    )

    adv_memory = RandomMemory(memory_size=args_cli.adv_rollouts, num_envs=env.num_envs, device=device)

    adv_cfg = PPO_CFG(
        rollouts=args_cli.adv_rollouts,
        learning_epochs=args_cli.adv_learning_epochs,
        mini_batches=args_cli.adv_mini_batches,
        learning_rate=args_cli.adv_lr,
        discount_factor=0.99,
        gae_lambda=0.95,
        grad_norm_clip=1.0,
        ratio_clip=0.2,
        value_clip=0.2,
        entropy_loss_scale=args_cli.adv_entropy_scale,
        value_loss_scale=2.0,
        learning_rate_scheduler=KLAdaptiveLR if args_cli.adv_kl_threshold > 0 else None,
        learning_rate_scheduler_kwargs=(
            {"kl_threshold": args_cli.adv_kl_threshold} if args_cli.adv_kl_threshold > 0 else {}
        ),
        state_preprocessor=RunningStandardScaler,
        state_preprocessor_kwargs={"size": adv_observation_space, "device": device},
        value_preprocessor=RunningStandardScaler,
        value_preprocessor_kwargs={"size": 1, "device": device},
        experiment={
            "directory": log_dir,
            "experiment_name": "adversary",
            "write_interval": "auto",
            "checkpoint_interval": "auto",
        },
    )
    adversary = PPO(
        models={"policy": adv_policy, "value": adv_value},
        memory=adv_memory,
        observation_space=adv_observation_space,
        state_space=adv_observation_space,
        action_space=adv_action_space,
        device=device,
        cfg=adv_cfg,
    )
    if args_cli.adv_checkpoint:
        print(f"[INFO] Loading adversary checkpoint from: {retrieve_file_path(args_cli.adv_checkpoint)}")
        adversary.load(retrieve_file_path(args_cli.adv_checkpoint))

    timesteps = agent_cfg["trainer"]["timesteps"]
    protagonist_rollouts = agent_cfg["agent"]["rollouts"]
    adversary.init(trainer_cfg=SequentialTrainerCfg(timesteps=timesteps, headless=True))
    dump_yaml(
        os.path.join(log_dir, "params", "adversary.yaml"),
        {
            "constraint_epsilon": args_cli.constraint_epsilon,
            "adv_lr": args_cli.adv_lr,
            "adv_rollouts": args_cli.adv_rollouts,
            "adv_learning_epochs": args_cli.adv_learning_epochs,
            "adv_mini_batches": args_cli.adv_mini_batches,
            "adv_hidden_units": args_cli.adv_hidden_units,
            "adv_entropy_scale": args_cli.adv_entropy_scale,
            "adv_warmup_iterations": args_cli.adv_warmup_iterations,
        },
    )

    # --- enable training mode: gates memory recording AND update scheduling for BOTH
    # agents (see skrl Agent.post_interaction/record_transition -- both no-op unless
    # `self.training` is True). SequentialTrainer.train() normally does this for us;
    # since we bypass it entirely to drive the loop by hand, we must do it ourselves. ---
    protagonist.enable_training_mode(True)
    adversary.enable_training_mode(True)

    start_time = time.time()

    # --- manual training loop: mirrors skrl/trainers/torch/base.py::Trainer.train(),
    # the loop MAPPO normally runs under SequentialTrainer, with the adversary spliced
    # in between protagonist.act() and env.step(). ---
    observations, infos = env.reset()
    states = env.state()

    for t in tqdm.tqdm(range(timesteps), file=sys.stdout):
        protagonist.pre_interaction(timestep=t, timesteps=timesteps)
        adversary.pre_interaction(timestep=t, timesteps=timesteps)

        with torch.no_grad():
            clean_actions, _ = protagonist.act(observations, states, timestep=t, timesteps=timesteps)

            adv_state = next(iter(states.values()))
            adv_action, _ = adversary.act(adv_state, adv_state, timestep=t, timesteps=timesteps)
            # During warmup, zero out only the perturbation actually APPLIED to the
            # environment -- never the action recorded for the adversary's own PPO
            # update. adv_action's log_prob was cached by adversary.act() for the action
            # it actually sampled; recording a zeroed-out action instead (as an earlier
            # version of this script did) would desync record_transition()'s `actions`
            # from that cached log_prob, corrupting the adversary's own importance-ratio
            # the same way the protagonist's clean/perturbed split (see below) is
            # carefully designed to avoid.
            if t < args_cli.adv_warmup_iterations * protagonist_rollouts:
                applied_adv_action = torch.zeros_like(adv_action)
            else:
                applied_adv_action = adv_action

            # split the flat adversary perturbation into per-agent slices (generic across tasks)
            perturbed_actions = {}
            offset = 0
            for uid in possible_agents:
                dim = int(env.action_spaces[uid].shape[0])
                perturbed_actions[uid] = clean_actions[uid] + applied_adv_action[:, offset : offset + dim]
                offset += dim

            next_observations, rewards, terminated, truncated, infos = env.step(perturbed_actions)
            next_states = env.state()

            # Clip rewards before either agent's advantage computation sees them. A rare
            # large outlier can collapse PPO's advantage-normalization std and explode the
            # policy loss in one update; this bounds both the protagonist's reward and (via
            # adv_reward below, derived from the clipped values) the adversary's.
            if args_cli.reward_clip > 0:
                rewards = {uid: r.clamp(-args_cli.reward_clip, args_cli.reward_clip) for uid, r in rewards.items()}

            # protagonist records its CLEAN action (matches its own cached log_prob) against
            # the reward from the PERTURBED step -- see module docstring.
            protagonist.record_transition(
                observations=observations,
                states=states,
                actions=clean_actions,
                rewards=rewards,
                next_observations=next_observations,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                infos=infos,
                timestep=t,
                timesteps=timesteps,
            )

            # zero-sum adversary reward; both tasks give an identical shared reward per
            # agent (cooperative), so mean vs. sum only rescales the signal.
            adv_reward = -sum(rewards.values()) / len(rewards)
            adv_terminated = torch.stack(list(terminated.values()), dim=0).any(dim=0)
            adv_truncated = torch.stack(list(truncated.values()), dim=0).any(dim=0)
            next_adv_state = next(iter(next_states.values()))
            adversary.record_transition(
                observations=adv_state,
                states=adv_state,
                actions=adv_action,
                rewards=adv_reward,
                next_observations=next_adv_state,
                next_states=next_adv_state,
                terminated=adv_terminated,
                truncated=adv_truncated,
                infos=infos,
                timestep=t,
                timesteps=timesteps,
            )

            # log environment info (mirrors base Trainer.train())
            env_info_key = agent_cfg["trainer"].get("environment_info", "log")
            if env_info_key in infos:
                for k, v in infos[env_info_key].items():
                    if isinstance(v, torch.Tensor) and v.numel() == 1:
                        protagonist.track_data(k if "/" in k else f"Info / {k}", v.item())

        protagonist.post_interaction(timestep=t, timesteps=timesteps)
        adversary.post_interaction(timestep=t, timesteps=timesteps)

        # NaN-guard: only cheap to check right after each agent's own update cadence,
        # since weights only change at those boundaries.
        if (t + 1) % protagonist_rollouts == 0:
            _assert_agent_finite(protagonist, "protagonist", t)
        if (t + 1) % args_cli.adv_rollouts == 0:
            _assert_agent_finite(adversary, "adversary", t)

        observations, states = next_observations, next_states

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
