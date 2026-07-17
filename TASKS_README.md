# Isaac Lab MARL Tasks

This document describes the two multi-agent reinforcement learning environments used in this project, trained with MAPPO via the SKRL framework.

| Task | Agents | Obs dim / agent | Action dim / agent | Algorithm |
|------|--------|-----------------|-------------------|-----------|
| Cart Double Pendulum | 2 | 4 / 3 | 1 / 1 | MAPPO |
| Shadow Hand Over | 2 | 157 | 20 | MAPPO |

---

## 1. Cart Double Pendulum

**Source:** [cart_double_pendulum_env.py](source/isaaclab_tasks/isaaclab_tasks/direct/cart_double_pendulum/cart_double_pendulum_env.py)  
**Agent config:** [agents/skrl_mappo_cfg.yaml](source/isaaclab_tasks/isaaclab_tasks/direct/cart_double_pendulum/agents/skrl_mappo_cfg.yaml)

### Overview

A two-link inverted pendulum mounted on a motorized cart. Two decentralized agents cooperate to balance both links upright: one controls the cart's horizontal translation, the other controls the torque at the second joint.

```
         ┌──── pendulum (link 2)
         │
    ─────┘ pole (link 1)
    │
[cart] ──────────────── rail
```

### State Space

Each agent receives a partial, local observation. All angles are normalized to `[−π, π]`.

| Agent | Dim | Elements |
|-------|-----|----------|
| **Cart** | 4 | `cart_pos` (m), `cart_vel` (m/s), `pole_pos` (rad), `pole_vel` (rad/s) |
| **Pendulum** | 3 | `pole_pos + pendulum_pos` (rad), `pendulum_pos` (rad), `pendulum_vel` (rad/s) |

- `pole_pos` — angle of link 1 relative to vertical (0 = upright)
- `pendulum_pos` — angle of link 2 relative to link 1
- The pendulum agent uses the **composite angle** `pole_pos + pendulum_pos` as its primary signal, which equals the absolute angle of the tip from vertical

### Action Space

Each agent outputs a single continuous scalar, clipped to `[−1, 1]` by the policy and then scaled:

| Agent | Dim | Applied to | Scale |
|-------|-----|------------|-------|
| **Cart** | 1 | `slider_to_cart` DOF (horizontal force) | × 100.0 N |
| **Pendulum** | 1 | `pole_to_pendulum` DOF (joint torque) | × 50.0 Nm |

### Goal of the Optimal Policy

Balance both links simultaneously in the upright position (all angles → 0) while keeping the cart within the rail bounds. An episode terminates early on failure:

- `|cart_pos| > 3.0 m` — cart out of bounds
- `|pole_pos| > π/2` — pole fallen past horizontal

Maximum episode length: **5 seconds** (600 steps at 120 Hz, decimation 2).

### Reward Design

Rewards are computed every step and distributed per agent. Both agents share the alive/termination signal; angle and velocity penalties are agent-specific.

| Term | Formula | Weight | Agent |
|------|---------|--------|-------|
| Alive bonus | `1 − terminated` | +1.0 | Both |
| Termination penalty | `terminated` | −2.0 | Both |
| Pole angle | `pole_pos²` | −1.0 | Cart |
| Cart velocity | `|cart_vel|` | −0.01 | Cart |
| Pole velocity | `|pole_vel|` | −0.01 | Cart |
| Pendulum tip angle | `(pole_pos + pendulum_pos)²` | −1.0 | Pendulum |
| Pendulum velocity | `|pendulum_vel|` | −0.01 | Pendulum |

**Cart agent total:**
```
R_cart = 1·(1−done) − 2·done − 1·pole_pos² − 0.01·|cart_vel| − 0.01·|pole_vel|
```

**Pendulum agent total:**
```
R_pendulum = 1·(1−done) − 2·done − 1·(pole_pos + pendulum_pos)² − 0.01·|pendulum_vel|
```

The squared-angle penalty grows quadratically with deviation — small disturbances are tolerated but large deflections are heavily penalized. Velocity penalties add light damping to discourage oscillation.

### Evaluation Video

<video controls width="800" src="logs/skrl/cart_double_pendulum_direct/2026-06-22_22-02-46_mappo_torch/videos/eval/rl-video-step-0.mp4">
  <a href="logs/skrl/cart_double_pendulum_direct/2026-06-22_22-02-46_mappo_torch/videos/eval/rl-video-step-0.mp4">Download evaluation video</a>
</video>

---

## 2. Shadow Hand Over

**Source:** [shadow_hand_over_env.py](source/isaaclab_tasks/isaaclab_tasks/direct/shadow_hand_over/shadow_hand_over_env.py)  
**Agent config:** [agents/skrl_mappo_convergence_cfg.yaml](source/isaaclab_tasks/isaaclab_tasks/direct/shadow_hand_over/agents/skrl_mappo_convergence_cfg.yaml)

### Overview

Two Shadow Dexterous Hands — mounted facing each other — must cooperatively grasp a small sphere and transfer it from the right hand to the left hand, reaching a randomized target pose. This is a fully cooperative task: both agents receive the same reward signal.

```
right hand ──► [sphere] ──► left hand
  (0, 0, 0.5)               (0, −1.0, 0.5)
```

Object starts near `(0, −0.39, 0.54)`. Target pose is randomized each episode.

### State Space

Each agent receives a **157-dimensional** observation vector. Both agents also observe the shared object and goal state.

| Group | Dim | Description |
|-------|-----|-------------|
| DOF positions | 24 | All 24 joint positions unscaled to `[lower, upper]` limits |
| DOF velocities | 24 | All 24 joint velocities, scaled by `vel_obs_scale = 0.2` |
| Fingertip positions | 15 | Cartesian XYZ of 5 fingertips (ffdistal, mfdistal, rfdistal, lfdistal, thdistal) |
| Fingertip rotations | 20 | Quaternion (4) per fingertip × 5 |
| Fingertip velocities | 20 | Linear (3) + angular (3) velocity per fingertip × 5 = 30 → padded to 20 |
| Previous actions | 20 | Last policy output (position targets) |
| Object position | 3 | Absolute XYZ of the sphere |
| Object rotation | 4 | Quaternion of the sphere |
| Object linear velocity | 3 | Sphere linear velocity |
| Object angular velocity | 3 | Sphere angular velocity, scaled by `vel_obs_scale = 0.2` |
| Goal position | 3 | Target XYZ |
| Goal rotation | 4 | Target quaternion |
| Goal–object rot. diff | 4 | `quat_mul(object_rot, quat_conjugate(goal_rot))` |
| **Total** | **157** | |

Each agent uses its own hand's DOF and fingertip data, but both see the same object and goal state.

**Global state for the value network (290-dim):** concatenation of both agents' 157-dim observations (with redundant shared fields kept for simplicity).

### Action Space

Each agent outputs a **20-dimensional** continuous vector of joint position targets, one per actuated DOF:

| Joint group | Joints | DOFs |
|-------------|--------|------|
| Wrist | WRJ1, WRJ0 | 2 |
| Fore finger | FFJ3, FFJ2, FFJ1 | 3 |
| Middle finger | MFJ3, MFJ2, MFJ1 | 3 |
| Ring finger | RFJ3, RFJ2, RFJ1 | 3 |
| Little finger | LFJ4, LFJ3, LFJ2, LFJ1 | 4 |
| Thumb | THJ4, THJ3, THJ2, THJ1, THJ0 | 5 |
| **Total** | | **20** |

Raw policy outputs `[−1, 1]` are scaled to each joint's `[lower, upper]` position limits. A moving-average filter (`act_moving_average = 1.0`, i.e. no smoothing) is applied before commanding the PD controller.

### Goal of the Optimal Policy

The two hands must learn a full manipulation pipeline cooperatively:

1. **Right hand** forms a grasp around the sphere at its initial position
2. **Both hands** coordinate to pass the sphere across the ~61 cm gap
3. **Left hand** receives the sphere and brings it to the randomized target pose

The optimal policy minimizes the distance between the sphere and the goal position at every timestep, so it must learn to move quickly and accurately.

### Reward Design

The reward function is intentionally minimal — a single dense term shared by both agents:

| Term | Formula | Agent |
|------|---------|-------|
| Distance reward | `2 × exp(−20 × ‖object_pos − goal_pos‖₂)` | Both |

```
goal_dist = ||object_pos − goal_pos||_2
rew = 2 · exp(−20 · goal_dist)
```

- The exponential shape gives a strong gradient when the object is close to the goal and decays quickly with distance, encouraging precision in the final placement.
- A scale of `dist_reward_scale = 20` means the reward drops to ~0.27 at 5 cm error and ~0.018 at 20 cm error.
- Maximum reward per step is **2.0** (object exactly at goal).
- Both agents receive identical reward — fully cooperative.

### Evaluation Videos

**Default evaluation view:**

<video controls width="800" src="logs/skrl/shadow_hand_over/2026-06-22_22-34-50_mappo_convergence_torch/videos/eval/rl-video-step-0.mp4">
  <a href="logs/skrl/shadow_hand_over/2026-06-22_22-34-50_mappo_convergence_torch/videos/eval/rl-video-step-0.mp4">Download eval video</a>
</video>

**Front view:**

<video controls width="800" src="logs/skrl/shadow_hand_over/2026-06-22_22-34-50_mappo_convergence_torch/videos/angle_front/rl-video-step-0.mp4">
  <a href="logs/skrl/shadow_hand_over/2026-06-22_22-34-50_mappo_convergence_torch/videos/angle_front/rl-video-step-0.mp4">Download front view video</a>
</video>

**Side view:**

<video controls width="800" src="logs/skrl/shadow_hand_over/2026-06-22_22-34-50_mappo_convergence_torch/videos/angle_side/rl-video-step-0.mp4">
  <a href="logs/skrl/shadow_hand_over/2026-06-22_22-34-50_mappo_convergence_torch/videos/angle_side/rl-video-step-0.mp4">Download side view video</a>
</video>

**Top view:**

<video controls width="800" src="logs/skrl/shadow_hand_over/2026-06-22_22-34-50_mappo_convergence_torch/videos/angle_top/rl-video-step-0.mp4">
  <a href="logs/skrl/shadow_hand_over/2026-06-22_22-34-50_mappo_convergence_torch/videos/angle_top/rl-video-step-0.mp4">Download top view video</a>
</video>

**Diagonal view:**

<video controls width="800" src="logs/skrl/shadow_hand_over/2026-06-22_22-34-50_mappo_convergence_torch/videos/angle_diagonal/rl-video-step-0.mp4">
  <a href="logs/skrl/shadow_hand_over/2026-06-22_22-34-50_mappo_convergence_torch/videos/angle_diagonal/rl-video-step-0.mp4">Download diagonal view video</a>
</video>

---

## 3. Diffusion Denoiser

**Scripts:** [collect_diffusion_data.py](scripts/reinforcement_learning/skrl/collect_diffusion_data.py), [train_diffusion.py](scripts/reinforcement_learning/skrl/train_diffusion.py), [eval_diffusion_sweep.py](scripts/reinforcement_learning/skrl/eval_diffusion_sweep.py), [run_diffusion_pipeline.sh](scripts/reinforcement_learning/skrl/run_diffusion_pipeline.sh)

### Overview

To make the trained MAPPO policies robust to corrupted actions (e.g. actuator noise, adversarial perturbation, communication error), each task has a DDPM-based *action denoiser*: a small MLP that predicts the noise ε added to a clean action and is used at evaluation time to reverse-diffuse a noisy action back toward the policy's clean action distribution before it is applied to the environment. Unlike a trajectory-level diffusion model, this denoiser operates **per timestep**: it is conditioned on the current global state only (not a fixed window of past states/actions), so there is no trajectory horizon `H` — every environment step is denoised independently.

### Architecture

`ActionDenoiser` — a plain MLP noise predictor, identical for both tasks except for width and I/O dims:

| Component    | Shape                                                      |
| ------------ | ----------------------------------------------------------- |
| `time_mlp`   | `Linear(1→H) → SiLU → Linear(H→H)`                          |
| `state_mlp`  | `Linear(Ds→H) → SiLU → Linear(H→H)`                          |
| `net`        | `Linear(Da+2H→H) → SiLU → Linear(H→H) → SiLU → Linear(H→Da)` |

The network takes the noisy action `x_t`, a normalized timestep embedding `t/T`, and a state embedding, and predicts the noise `ε` to remove. Forward diffusion follows a standard linear-beta DDPM schedule: `T = 100` steps, `β_start = 1e-4`, `β_end = 2e-2`.

| Task | Hidden dim H | State dim Ds | Action dim Da |
|---|---|---|---|
| Cart Double Pendulum | 256 | 14 | 2 |
| Shadow Hand Over | 512 | 580 | 40 |

(`Ds`/`Da` are the concatenated per-agent global state and joint action vectors used by the centralized MAPPO critic.)

### Training Data Collection

`collect_diffusion_data.py` rolls out a single trained MAPPO checkpoint (best-reward agent, deterministic `mean_actions`, no injected noise) for **500 episodes across 64 parallel environments**, recording the `(state, action)` pair at every step into a flat buffer (episode boundaries are not preserved — each transition is an independent training sample). Collected dataset sizes:

| Task | Samples | States shape | Actions shape |
|---|---|---|---|
| Cart Double Pendulum | 153,088 | `(153088, 14)` | `(153088, 2)` |
| Shadow Hand Over | 229,888 | `(229888, 580)` | `(229888, 40)` |

### Training

The denoiser is trained on these `(state, action)` pairs with a forward diffusion process of `T = 100` steps (linear schedule, `β_start = 1e-4`, `β_end = 2e-2`). Training uses a batch size of 512 over 100 epochs, optimized with Adam at a learning rate of `1 × 10⁻⁴`, minimizing MSE between predicted and true injected noise. States and actions are z-score normalized beforehand (mean/std stored in the checkpoint for inference-time un-normalization). Each task gets its own denoiser trained from a single policy's rollout data — this is a per-timestep, state-conditioned model, not a fixed-horizon trajectory diffusion model.

### Evaluation Protocol

`eval_diffusion_sweep.py` evaluates robustness under injected Gaussian action noise, with and without denoising, for **200 episodes × 64 environments per configuration**. Per step: the policy produces a clean action → Gaussian noise `𝒩(0,1)·noise_std + noise_mean` is added → the noisy action is optionally passed through the trained denoiser's reverse diffusion process, run from a configurable starting step `t_start` down to `0` (the noisy action is treated directly as `x_{t_start}`; larger `t_start` means more reverse-diffusion steps and thus stronger denoising) → the resulting action is applied via `env.step`. Two sweeps are run per task: a **noise-std sweep** (`t_start ∈ {20, 40, 60}` vs. no denoising, at `noise_std = 0` as the clean baseline) and a **noise-mean sweep** (`noise_mean ∈ {-1.0, -0.5, 0.0, 0.5, 1.0}` crossed with a reduced std list). Results report mean cumulative reward ± std and a full-episode-rate (Cart) or success-rate (Shadow Hand, object within 0.1 m of goal) percentage, comparing `no_denoise` against `denoiser_t20`/`t40`/`t60`.

### Results — Noise-std Sweep

**Cart Double Pendulum** (`results/cart_diffusion_sweep.txt`, reward ± std [full-episode-rate %]):

| noise_std | no_denoise | denoiser_t20 | denoiser_t40 | denoiser_t60 |
|---|---|---|---|---|
| 0.0 | 294.6±4.0 [100%] | 294.6±3.9 [100%] | 294.8±3.8 [100%] | 294.9±3.6 [100%] |
| 1.0 | 259.9±5.3 [100%] | 294.2±5.4 [100%] | 294.1±5.3 [100%] | 294.1±4.7 [100%] |
| 5.0 | -25.0±93.0 [66.5%] | 258.5±79.9 [78.5%] | 263.4±66.2 [68.5%] | 251.3±76.8 [68.5%] |
| 10.0 | -89.9±92.5 [0.5%] | 245.4±96.4 [68.5%] | 259.3±72.9 [68.5%] | 260.5±65.0 [68.5%] |
| 20.0 | -49.3±49.1 [0.5%] | 261.2±84.9 [68.5%] | 276.1±46.4 [68.5%] | 219.9±110.8 [68.0%] |
| 30.0 | -32.7±26.3 [0.5%] | 260.8±87.3 [71.0%] | 273.7±50.6 [68.5%] | 215.8±117.3 [68.5%] |

**Shadow Hand Over** (`results/shadow_diffusion_sweep.txt`, reward ± std [success-rate %]):

| noise_std | no_denoise | denoiser_t20 | denoiser_t40 | denoiser_t60 |
|---|---|---|---|---|
| 0.0 | 842.1±2.2 [100%] | 842.3±2.0 [100%] | 842.1±2.2 [100%] | 842.2±2.0 [100%] |
| 1.0 | 814.0±57.2 [100%] | 829.7±95.9 [100%] | 813.9±144.2 [100%] | 812.9±138.9 [100%] |
| 5.0 | 498.5±295.7 [97.0%] | 729.9±250.7 [100%] | 728.3±252.6 [100%] | 728.9±245.7 [100%] |
| 10.0 | 264.5±299.7 [87.0%] | 694.3±276.0 [99.5%] | 712.7±260.2 [100%] | 728.4±238.3 [100%] |
| 20.0 | 127.0±213.2 [80.5%] | 689.2±288.2 [96.0%] | 717.1±254.0 [100%] | 745.4±216.5 [100%] |
| 30.0 | 121.2±196.0 [78.5%] | 648.9±334.8 [90.0%] | 674.6±310.7 [100%] | 699.9±288.6 [100%] |
| 50.0 | 23.8±53.4 [61.5%] | 573.5±387.4 [83.0%] | 598.2±366.0 [100%] | 628.9±350.2 [100%] |

In both tasks, denoising recovers most of the clean-policy reward and full-episode/success rate that noise alone destroys, with the benefit growing at higher noise magnitudes. Full noise-mean sweep tables (`noise_mean × noise_std`) are under [results/noise_mu_sweeps/](results/noise_mu_sweeps/).

### RARL-hardened Variant

The denoiser has also been evaluated on top of policies trained with the separate RARL-style action-adversarial method (`train_action_adv.py`, Pinto et al. 2017) — e.g. `results/noise_mu_sweeps/mappo_act_adv_eps1_diffusion_noise_mean_cart.txt` runs the same denoising sweep against a MAPPO checkpoint hardened with a learned adversary (`--constraint-epsilon 1.0`). This tests whether training-time adversarial robustness and eval-time diffusion denoising compound.
