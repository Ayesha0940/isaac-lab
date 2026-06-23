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

## 3. Action Noise Robustness

To stress-test policy robustness, zero-mean Gaussian noise is injected into the **policy action outputs** before they are passed to the environment, simulating actuator noise or control disturbances at deployment time.

### Noise Injection Mechanism

```
policy_output  →  + N(0, σ²)  →  env.step(noisy_action)
```

- Noise is applied **after** the policy's deterministic mean action is computed (`mean_actions`) and **before** `env.step()` is called.
- The same `σ` (noise std) is applied independently to every action dimension of every agent at every timestep.
- No manual clipping is applied — the environment handles it:
  - **Cart Double Pendulum**: actions are unbounded; noise passes through directly and is scaled by the physical constants (×100 N / ×50 Nm).
  - **Shadow Hand Over**: actions are scaled from `[−1, 1]` to joint limits via `scale()`, then hard-clipped to physical joint ranges via `saturate()`. This means extreme noise (`σ > 1`) is partially absorbed by joint-limit saturation.

The noise std `σ` is in the **normalized policy output space** (same units as the network's output, typically `~[−1, 1]` at convergence).

---

### Results: Cart Double Pendulum

**Success metric:** full-episode rate — fraction of 200 episodes where the cart/pendulum never fell (reached max episode length without early termination).

| noise σ | reward mean ± std | full-episode rate |
|---------|-------------------|-------------------|
| 0.00 | 294.47 ± 4.29 | 100.0% |
| 1.00 | 259.11 ± 4.98 | 100.0% |
| 2.00 | 215.51 ± 27.59 | 99.5% |
| **5.00** | **−15.31 ± 84.32** | **70.5%** |
| 10.00 | −87.55 ± 87.73 | 1.0% |
| 20.00 | −46.91 ± 41.02 | 0.5% |

**Key observations:**

- The policy is highly robust up to `σ = 2` — essentially zero degradation in episode completion despite a 27% reward drop.
- There is a sharp failure cliff between `σ = 2` and `σ = 5`. At `σ = 5`, noise std already exceeds the typical policy output magnitude (~1), making the applied force/torque largely random.
- At `σ ≥ 10`, the policy almost always fails immediately — the noise-induced forces (~1000 N random perturbation vs. ~100 N control signal) overwhelm the system.
- The negative mean reward at high noise arises from the termination penalty (−2 per episode) outweighing the alive bonus.

**Noise videos (front view):**

<video controls width="800" src="logs/skrl/cart_double_pendulum_direct/videos/noise_std_5/rl-video-step-0.mp4">
  <a href="logs/skrl/cart_double_pendulum_direct/videos/noise_std_5/rl-video-step-0.mp4">Download σ=5 video</a>
</video>

<video controls width="800" src="logs/skrl/cart_double_pendulum_direct/videos/noise_std_20/rl-video-step-0.mp4">
  <a href="logs/skrl/cart_double_pendulum_direct/videos/noise_std_20/rl-video-step-0.mp4">Download σ=20 video</a>
</video>

---

### Results: Shadow Hand Over

**Success metric:** success rate — fraction of 200 episodes where the object came within **5 cm** of the goal at any timestep. Success is inferred from the instantaneous reward via `dist = −ln(r / 2) / 20`.

| noise σ | reward mean ± std | success rate (< 5 cm) |
|---------|-------------------|----------------------|
| 0.00 | 842.14 ± 2.07 | 100.0% |
| 1.00 | 814.35 ± 57.31 | 99.5% |
| 2.00 | 775.79 ± 104.34 | 99.5% |
| 5.00 | 549.84 ± 306.12 | 86.0% |
| **10.00** | **285.01 ± 310.44** | **66.5%** |
| 20.00 | 77.67 ± 161.77 | 51.0% |
| 30.00 | 47.26 ± 97.67 | 32.0% |

**Key observations:**

- Shadow Hand is **substantially more robust** than Cart Double Pendulum. At `σ = 5` it still achieves 86% success, vs. only 70.5% full-episode rate for Cart.
- The robustness comes from two sources: (1) **heavy domain randomization** during training (gravity, joint stiffness, friction, mass randomized every episode) forces the policy to learn noise-tolerant behaviors; (2) **joint-limit saturation** in the environment clips extreme noise values, attenuating the effective disturbance.
- Degradation is gradual rather than cliff-like — the policy partially compensates even under large noise by leveraging redundancy in the 20-DOF hands.
- At `σ = 30`, success rate falls to 32% and mean reward to 47/900. The policy still occasionally completes the task but can no longer reliably do so.
- The large reward standard deviation at high noise (e.g., ±310 at `σ = 10`) reflects a bimodal outcome distribution: some episodes succeed fully while others fail completely, rather than all episodes degrading uniformly.

**Noise videos (front view):**

<video controls width="800" src="logs/skrl/shadow_hand_over/videos/noise_std_10/rl-video-step-0.mp4">
  <a href="logs/skrl/shadow_hand_over/videos/noise_std_10/rl-video-step-0.mp4">Download σ=10 video</a>
</video>

<video controls width="800" src="logs/skrl/shadow_hand_over/videos/noise_std_30/rl-video-step-0.mp4">
  <a href="logs/skrl/shadow_hand_over/videos/noise_std_30/rl-video-step-0.mp4">Download σ=30 video</a>
</video>

---

### Summary Comparison

| | Cart Double Pendulum | Shadow Hand Over |
|---|---|---|
| Robust up to σ ≈ | **2** | **5** |
| 50% success at σ ≈ | 4–5 | 20 |
| Failure mode | Sharp cliff (brittle balance task) | Gradual degradation (redundant DOFs) |
| Source of robustness | None (no domain rand.) | Joint saturation + domain randomization |
| Noise absorption | Direct (unbounded actions) | Partial (saturate() clips to joint limits) |

Raw results: [results/cart_noise_sweep_wide.txt](results/cart_noise_sweep_wide.txt) · [results/shadow_noise_sweep_wide.txt](results/shadow_noise_sweep_wide.txt)
