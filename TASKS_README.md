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
