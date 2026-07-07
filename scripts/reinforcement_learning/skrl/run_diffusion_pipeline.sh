#!/usr/bin/env bash
# Diffusion denoiser pipeline for Cart Double Pendulum and Shadow Hand Over.
#
# Three phases:
#   1. Collect clean (state, action) pairs from trained MAPPO policy (needs Isaac Sim)
#   2. Train DDPM action denoiser (plain Python, no Isaac Sim)
#   3. Evaluate robustness with vs without denoiser across noise levels (needs Isaac Sim)
#
# Usage:
#   bash scripts/reinforcement_learning/skrl/run_diffusion_pipeline.sh
#
# Outputs (all under results/):
#   diffusion_data_cart.npz          — collected trajectories (Cart DP)
#   diffusion_data_shadow.npz        — collected trajectories (Shadow Hand Over)
#   diffusion_model_cart.pt          — trained denoiser (Cart DP)
#   diffusion_model_shadow.pt        — trained denoiser (Shadow Hand Over)
#   cart_diffusion_sweep.txt         — eval table (Cart DP)
#   shadow_diffusion_sweep.txt       — eval table (Shadow Hand Over)

set -euo pipefail

ISAACLAB_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SCRIPTS="$ISAACLAB_ROOT/scripts/reinforcement_learning/skrl"
RESULTS="$ISAACLAB_ROOT/results"
mkdir -p "$RESULTS"

CART_CKPT="$ISAACLAB_ROOT/logs/skrl/cart_double_pendulum_direct/2026-06-22_22-02-46_mappo_torch/checkpoints/best_agent.pt"
SHADOW_CKPT="$ISAACLAB_ROOT/logs/skrl/shadow_hand_over/2026-06-22_22-34-50_mappo_convergence_torch/checkpoints/best_agent.pt"

NOISE_STDS="0 1 2 5 10 20 30 40 50"
T_START_LIST="20 40 60"
NUM_ENVS=64
NUM_EPISODES=200
COLLECT_EPISODES=500

# echo "============================================================"
# echo "  Diffusion Denoiser Pipeline"
# echo "  ISAACLAB_ROOT: $ISAACLAB_ROOT"
# echo "  noise_stds: $NOISE_STDS"
# echo "  t_start_list: $T_START_LIST"
# echo "============================================================"

# # ============================================================
# # PHASE 1: Data collection
# # ============================================================
# echo ""
# echo "[Phase 1/3] Collecting expert trajectories ..."

# echo ""
# echo "  [1a] Cart Double Pendulum"
# conda run -n isaaclab bash "$ISAACLAB_ROOT/isaaclab.sh" \
#   -p "$SCRIPTS/collect_diffusion_data.py" \
#   --task Isaac-Cart-Double-Pendulum-Direct-v0 \
#   --algorithm MAPPO \
#   --checkpoint "$CART_CKPT" \
#   --num_envs "$NUM_ENVS" \
#   --num_episodes "$COLLECT_EPISODES" \
#   --output "$RESULTS/diffusion_data_cart.npz" \
#   --headless

# echo ""
# echo "  [1b] Shadow Hand Over"
# conda run -n isaaclab bash "$ISAACLAB_ROOT/isaaclab.sh" \
#   -p "$SCRIPTS/collect_diffusion_data.py" \
#   --task Isaac-Shadow-Hand-Over-Direct-v0 \
#   --algorithm MAPPO \
#   --agent skrl_mappo_convergence_cfg_entry_point \
#   --checkpoint "$SHADOW_CKPT" \
#   --num_envs "$NUM_ENVS" \
#   --num_episodes "$COLLECT_EPISODES" \
#   --output "$RESULTS/diffusion_data_shadow.npz" \
#   --headless

# # ============================================================
# # PHASE 2: Denoiser training (plain Python, no Isaac Sim)
# # ============================================================
# echo ""
# echo "[Phase 2/3] Training diffusion denoisers ..."

# echo ""
# echo "  [2a] Cart Double Pendulum (hidden_dim=256)"
# conda run -n isaaclab python "$SCRIPTS/train_diffusion.py" \
#   --data_path "$RESULTS/diffusion_data_cart.npz" \
#   --output "$RESULTS/diffusion_model_cart.pt" \
#   --hidden_dim 256 \
#   --diffusion_steps 100 \
#   --epochs 100 \
#   --batch_size 512

# echo ""
# echo "  [2b] Shadow Hand Over (hidden_dim=512)"
# conda run -n isaaclab python "$SCRIPTS/train_diffusion.py" \
#   --data_path "$RESULTS/diffusion_data_shadow.npz" \
#   --output "$RESULTS/diffusion_model_shadow.pt" \
#   --hidden_dim 512 \
#   --diffusion_steps 100 \
#   --epochs 100 \
#   --batch_size 512

# # ============================================================
# # PHASE 3: Robustness evaluation
# # ============================================================
# echo ""
# echo "[Phase 3/3] Running diffusion denoiser sweep ..."

# echo ""
# echo "  [3a] Cart Double Pendulum (success = full-episode rate)"
# conda run -n isaaclab bash "$ISAACLAB_ROOT/isaaclab.sh" \
#   -p "$SCRIPTS/eval_diffusion_sweep.py" \
#   --task Isaac-Cart-Double-Pendulum-Direct-v0 \
#   --algorithm MAPPO \
#   --checkpoint "$CART_CKPT" \
#   --diffusion_model "$RESULTS/diffusion_model_cart.pt" \
#   --noise_stds $NOISE_STDS \
#   --t_start_list $T_START_LIST \
#   --num_envs "$NUM_ENVS" \
#   --num_episodes "$NUM_EPISODES" \
#   --out "$RESULTS/cart_diffusion_sweep.txt" \
#   --headless

echo ""
echo "  [3b] Shadow Hand Over (success = object within 0.1 m of goal)"
conda run -n isaaclab bash "$ISAACLAB_ROOT/isaaclab.sh" \
  -p "$SCRIPTS/eval_diffusion_sweep.py" \
  --task Isaac-Shadow-Hand-Over-Direct-v0 \
  --algorithm MAPPO \
  --agent skrl_mappo_convergence_cfg_entry_point \
  --checkpoint "$SHADOW_CKPT" \
  --diffusion_model "$RESULTS/diffusion_model_shadow.pt" \
  --noise_stds $NOISE_STDS \
  --t_start_list $T_START_LIST \
  --num_envs "$NUM_ENVS" \
  --num_episodes "$NUM_EPISODES" \
  --success_threshold 0.1 \
  --reward_scale 2.0 \
  --dist_reward_scale 20.0 \
  --out "$RESULTS/shadow_diffusion_sweep.txt" \
  --headless

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "  RESULTS"
echo "============================================================"
echo ""
echo "=== Cart Double Pendulum ==="
cat "$RESULTS/cart_diffusion_sweep.txt"
echo ""
echo "=== Shadow Hand Over ==="
cat "$RESULTS/shadow_diffusion_sweep.txt"
