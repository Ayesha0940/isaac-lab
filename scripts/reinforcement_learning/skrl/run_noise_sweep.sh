#!/usr/bin/env bash
# Gaussian action noise robustness sweep for Cart Double Pendulum and Shadow Hand Over.
# Runs two Isaac Sim sessions sequentially (one per task), each evaluating all noise levels.
#
# Usage:
#   bash scripts/reinforcement_learning/skrl/run_noise_sweep.sh
#
# Outputs:
#   /tmp/cart_noise_sweep.txt     — Cart Double Pendulum results
#   /tmp/shadow_noise_sweep.txt   — Shadow Hand Over results

set -euo pipefail

ISAACLAB_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SCRIPT="$ISAACLAB_ROOT/scripts/reinforcement_learning/skrl/eval_noise_sweep.py"

CART_CKPT="$ISAACLAB_ROOT/logs/skrl/cart_double_pendulum_direct/2026-06-22_22-02-46_mappo_torch/checkpoints/best_agent.pt"
SHADOW_CKPT="$ISAACLAB_ROOT/logs/skrl/shadow_hand_over/2026-06-22_22-34-50_mappo_convergence_torch/checkpoints/best_agent.pt"

NOISE_STDS="0 0.5 1.0 1.5 2.0"
NUM_ENVS=64
NUM_EPISODES=200

echo "============================================================"
echo "  Noise Robustness Sweep"
echo "  noise_stds: $NOISE_STDS"
echo "  num_envs=$NUM_ENVS  num_episodes=$NUM_EPISODES"
echo "============================================================"

# --- Task 1: Cart Double Pendulum ---
echo ""
echo "[1/2] Cart Double Pendulum (success = full-episode rate)"
conda run -n isaaclab bash "$ISAACLAB_ROOT/isaaclab.sh" \
  -p "$SCRIPT" \
  --task Isaac-Cart-Double-Pendulum-Direct-v0 \
  --algorithm MAPPO \
  --checkpoint "$CART_CKPT" \
  --noise_stds $NOISE_STDS \
  --num_envs $NUM_ENVS \
  --num_episodes $NUM_EPISODES \
  --headless \
  --out "$ISAACLAB_ROOT/results/cart_noise_sweep.txt"

# --- Task 2: Shadow Hand Over ---
echo ""
echo "[2/2] Shadow Hand Over (success = object within 0.1 m of goal)"
conda run -n isaaclab bash "$ISAACLAB_ROOT/isaaclab.sh" \
  -p "$SCRIPT" \
  --task Isaac-Shadow-Hand-Over-Direct-v0 \
  --algorithm MAPPO \
  --agent skrl_mappo_convergence_cfg_entry_point \
  --checkpoint "$SHADOW_CKPT" \
  --noise_stds $NOISE_STDS \
  --num_envs $NUM_ENVS \
  --num_episodes $NUM_EPISODES \
  --success_threshold 0.1 \
  --reward_scale 2.0 \
  --dist_reward_scale 20.0 \
  --headless \
  --out "$ISAACLAB_ROOT/results/shadow_noise_sweep.txt"

# --- Combined summary ---
echo ""
echo "============================================================"
echo "  RESULTS"
echo "============================================================"
echo ""
echo "=== Cart Double Pendulum ==="
cat "$ISAACLAB_ROOT/results/cart_noise_sweep.txt"

echo ""
echo "=== Shadow Hand Over ==="
cat "$ISAACLAB_ROOT/results/shadow_noise_sweep.txt"
