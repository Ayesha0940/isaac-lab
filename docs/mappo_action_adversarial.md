# MAPPO Action-Adversarial Training

Robust MARL setup implemented in
`scripts/reinforcement_learning/skrl/train_action_adv.py`.

---

## 1. Motivation

A vanilla MAPPO policy is trained in a clean environment and evaluated with
adversarial noise. The policy has never seen perturbations, so it is brittle:
even small biased noise (e.g. μ = −1, σ = 0) can collapse reward by 400+ points
on the Cart Double Pendulum task.

Action-adversarial training (RARL — Pinto et al. 2017) exposes the protagonist
to worst-case perturbations *during training*, forcing the policy to become
inherently robust rather than relying on a post-hoc denoiser at eval time.

---

## 2. Tasks

| Task | Protagonist agents | Joint action dim Da | Global state dim Ds |
|---|---|---|---|
| Cart Double Pendulum | cart (1), pendulum (1) | 2 | 7 |
| Shadow Hand Over | right_hand (20), left_hand (20) | 40 | 290 |

---

## 3. Protagonist — MAPPO

### 3.1 Algorithm

Multi-Agent PPO with **centralized training, decentralized execution (CTDE)**:

- Each agent has its own **policy network** (actor) that sees only its local
  observation.
- Each agent has a **value network** (critic) that sees the global state from
  `env.state()` — this is the "centralized" part.
- During execution, only the actors are used (decentralized).

### 3.2 Architecture

Configured in `agents/skrl_mappo_cfg.yaml` per task:

**Cart Double Pendulum:**
```
Policy (per agent):  Linear(obs_dim → 32) → ELU → Linear(32 → 32) → ELU → Linear(32 → action_dim)
Value  (per agent):  Linear(state_dim → 32) → ELU → Linear(32 → 32) → ELU → Linear(32 → 1)
obs_dim:    cart=4, pendulum=3
state_dim:  7 (shared global state)
```

**Shadow Hand Over:**
```
Policy (per agent):  Linear(157 → 512) → ELU → Linear(512 → 512) → ELU → ...
                     → Linear(256 → 128) → ELU → Linear(128 → 20)
Value  (per agent):  same depth, input=state_dim (290)
```

Both use a **Gaussian policy head**: the network outputs mean μ and a learned
log-std parameter σ; actions are sampled as `a ~ N(μ, σ²)` during training and
taken as `μ` at eval time (deterministic mean).

### 3.3 Training Objective

Standard PPO, run identically for each protagonist agent:

```
L_PPO = L_clip + c_v · L_value + c_e · L_entropy

L_clip  = E[min(r_t · A_t, clip(r_t, 1-ε, 1+ε) · A_t)]
            where r_t = π(a|o) / π_old(a|o)  (importance ratio)

L_value = E[(V(s) - V_target)²]              (with optional value clipping)

L_entropy = E[-π log π]                       (exploration bonus)

A_t  = GAE(λ) advantage estimate
```

Key hyperparameters (Cart DP defaults from yaml):
```
learning_rate:      3e-4 (KL-adaptive scheduler)
rollouts:           16
learning_epochs:    8
mini_batches:       1
discount_factor:    0.99
gae_lambda:         0.95
ratio_clip:         0.2
grad_norm_clip:     1.0
entropy_loss_scale: 0.0
```

---

## 4. Adversary — PPO

### 4.1 Role

A **single** PPO agent that operates at the joint level: it observes the global
state and outputs one perturbation vector covering all protagonist agents'
actions simultaneously.

### 4.2 Observation & Action Space

```
obs:    global state  [Ds]       (same signal the protagonist's critic sees)
action: perturbation  [Da]       (one entry per protagonist action dimension)
        clipped to [-ε, +ε] elementwise   (constraint_epsilon, default 0.1)
```

The perturbation space is a `Box(low=-ε, high=ε, shape=(Da,))` — the adversary's
actions are clipped to this box both by the Gaussian policy's `clip_actions=True`
flag and by the box bounds, so the constraint is hard.

### 4.3 Architecture

```
Policy: Linear(Ds → H) → ELU → Linear(H → H) → ELU → Linear(H → Da)
Value:  Linear(Ds → H) → ELU → Linear(H → H) → ELU → Linear(H → 1)

Default H = [64, 64]    (--adv-hidden-units, much smaller than protagonist)
```

Gaussian policy head with:
```
log_std:      learned parameter, clamped to [-20, 2]
initial_std:  1.0 (= exp(0.0))
```

State preprocessor: `RunningStandardScaler` on the global state (online mean/std).
Value preprocessor: `RunningStandardScaler` on the scalar value target.

### 4.4 Training Objective

The adversary maximizes the **negative** protagonist reward — zero-sum:

```
adv_reward = -mean(protagonist_rewards_per_agent)

L_adv = PPO_objective(adv_reward)
      = L_clip(adv_reward) + 2.0 · L_value + 0.0 · L_entropy
```

The adversary is a standard skrl `PPO` agent; its objective is identical in
form to the protagonist's, just operating on negated cooperative rewards.

### 4.5 Key Hyperparameters (defaults)

```
adv_lr:               3e-4
adv_rollouts:         16
adv_learning_epochs:  8
adv_mini_batches:     1
adv_entropy_scale:    0.0
constraint_epsilon:   0.1
adv_warmup_iterations: 0
```

---

## 5. Training Loop Mechanics

This is the most subtle part of the implementation, and the deviation from a
naive "perturb then record" approach.

### 5.1 The Clean/Perturbed Split

On-policy PPO caches the log-probability `log π(a|o)` at the moment of sampling
inside `agent.act()`. The surrogate loss later computes the importance ratio
`π_new(a|o) / π_old(a|o)`, which requires that `a` fed into `record_transition()`
is **exactly** the action that was sampled (the one whose log-prob was cached).

If the protagonist recorded the *perturbed* action but cached the log-prob of the
*clean* action, the importance ratio would be `π_new(a_perturbed | o) / π_old(a_clean | o)` — invalid.

The fix, following Pinto et al. 2017:

```
protagonist.act(obs)           → clean_action  (log-prob cached for clean_action)
adversary.act(state)           → perturbation
env.step(clean_action + δ)     → next_obs, reward          ← perturbed step
protagonist.record_transition(actions=clean_action, reward=reward)
                                                            ← clean action, perturbed reward
adversary.record_transition(actions=perturbation, reward=-reward)
```

The protagonist trains to maximize reward in a world where its clean actions are
perturbed by an adversary. The adversary trains to find the worst perturbation.
Both use PPO with valid importance ratios.

### 5.2 Warmup Phase

`--adv-warmup-iterations N` zeros out only the perturbation *applied to the
environment* (not the one recorded for the adversary's PPO update) for the first
`N × protagonist_rollouts` timesteps. This lets the protagonist establish a
baseline policy before the adversary starts attacking.

The adversary still samples and records transitions during warmup (to bootstrap
its own value estimates), but receives zero-reward environment feedback — its
policy converges toward the null perturbation before gradually learning to attack.

### 5.3 Update Interleaving

Both agents call `pre_interaction` → `record_transition` → `post_interaction` at
every environment step. `post_interaction` triggers a PPO gradient update
internally when the agent's memory is full (i.e., every `rollouts` steps). Since
both agents use `rollouts=16` by default, they update at the same frequency and
on the same environment interactions — a fully interleaved alternating update.

### 5.4 Per-Step Diagram

```
t=0,1,...,timesteps:
  ┌─────────────────────────────────────────────────────────────────┐
  │ obs, state ← env                                               │
  │                                                                 │
  │ clean_action ← protagonist.act(obs, state)  [log-prob cached]  │
  │ δ            ← adversary.act(state)         [log-prob cached]  │
  │ δ_applied    = δ  (or 0 during warmup)                         │
  │                                                                 │
  │ perturbed_action = clean_action + δ_applied                    │
  │ next_obs, reward ← env.step(perturbed_action)                  │
  │                                                                 │
  │ protagonist.record(actions=clean_action, reward=reward)         │
  │ adversary.record(actions=δ,             reward=-reward)         │
  │                                                                 │
  │ protagonist.post_interaction()  → PPO update every 16 steps    │
  │ adversary.post_interaction()    → PPO update every 16 steps    │
  └─────────────────────────────────────────────────────────────────┘
```

---

## 6. NaN Guard

After every PPO update (`t % rollouts == 0`), all policy and value parameter
tensors are scanned for `NaN` / `Inf`. If found, training aborts immediately with
a diagnostic message. This catches adversary divergence (which can happen with
large ε or high adv_lr) within one rollout rather than silently burning the rest
of the run.

---

## 7. Empirical Results

### Cart Double Pendulum (ε = 1.0, 200 episodes × 64 envs)

Metric is **full-episode rate** (fraction reaching max episode length = task success).

| noise_mean | noise_std | Vanilla MAPPO reward | Adv-MAPPO reward | Improvement |
|---|---|---|---|---|
| −1.0 | 0.0 | −251 | **+168** | +419 |
| −1.0 | 0.5 | −193 | **+242** | +435 |
| −1.0 | 1.0 | −201 | **+233** | +434 |
| 0.0 | 0.0 | +245 | +187 | −58 (small regression) |
| 0.0 | 1.0 | +214 | **+228** | +14 |
| +1.0 | 1.0 | +147 | **+233** | +86 |

The adversarially-trained policy is dramatically more robust to biased (non-zero
mean) noise, at a modest cost to clean-environment performance (−58 reward at
μ=0, σ=0). The full-episode rate stabilises around 68–69% across the entire
noise sweep, versus near-zero rate for vanilla MAPPO at mean-shifted noise.

### Shadow Hand Over (ε = 0.1 default)

The adversarially-trained policy maintains full-episode rate (100%) only at
μ = −1, σ ≤ 0.5. All other noise configurations collapse (nan rewards, ~1%
full-ep rate). This indicates the adversary's perturbation budget (ε = 0.1)
was too small relative to the noise magnitudes being evaluated (σ up to 2.0),
and the 40-dim joint action space makes the robustness problem harder.

---

## 8. Limitations and Design Notes

**Log-prob mismatch risk (MAPPO-specific):** MAPPO's centralized critic takes the
global state and all agents' actions; however the surrogate objective only depends
on the per-agent importance ratio `π_new(a_i|o_i) / π_old(a_i|o_i)`. Recording
the clean action preserves this ratio correctly.

**ε must cover the expected eval noise range:** The adversary can only apply
perturbations bounded by ε. If eval noise std > ε, the trained robustness does
not transfer. For Cart DP, ε = 1.0 was sufficient to cover eval stds up to 2.0.
For Shadow Hand Over, ε should be increased (try ε = 0.5–1.0).

**Protagonist pays a clean-performance tax:** The policy learns to tolerate
worst-case perturbations, which slightly reduces peak performance on clean inputs.
This is the standard robustness/performance trade-off.

**Adversary update frequency:** Both agents currently update every 16 steps
(same rollout length). Increasing the adversary's rollout length (e.g. 32)
would give it more gradient steps relative to the protagonist, making it a
stronger adversary — useful if the protagonist adapts too quickly.

**No curriculum:** ε is fixed throughout training. A curriculum that ramps ε
from 0 → target over the first N% of training can stabilise early learning,
especially for Shadow Hand Over.
