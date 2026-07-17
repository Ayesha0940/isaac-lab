#!/usr/bin/env bash
# Action-delay fault robustness sweep across three checkpoint types
# (MAPPO baseline, IPPO baseline, MAPPO action-adversarially-trained), each
# also evaluated with the trained diffusion denoiser, for Cart Double Pendulum
# and Shadow Hand Over. The denoiser was trained only on vanilla-MAPPO
# trajectories, so its IPPO/act-adv runs are a cross-policy generalization test.
#
# Usage:
#   bash scripts/reinforcement_learning/skrl/run_delay_sweep.sh
#
# Outputs:
#   results/{mappo,ippo_tuned,mappo_act_adv}_delay_sweep_cart.txt
#   results/{mappo,ippo,mappo_act_adv}_delay_sweep_shadow.txt
#   results/delay_sweep/cart/{mappo,ippo_tuned,mappo_act_adv}_diffusion_delay_sweep_cart.txt
#   results/delay_sweep/shadow/{mappo,ippo,mappo_act_adv}_diffusion_delay_sweep_shadow.txt

set -euo pipefail

ISAACLAB_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SCRIPT="$ISAACLAB_ROOT/scripts/reinforcement_learning/skrl/eval_noise_sweep.py"
DIFFUSION_SCRIPT="$ISAACLAB_ROOT/scripts/reinforcement_learning/skrl/eval_diffusion_sweep.py"

DELAY_K_LIST="0 1 2 3 5 8"
NUM_ENVS=64
NUM_EPISODES=200

CART_TASK="Isaac-Cart-Double-Pendulum-Direct-v0"
CART_TAGS=(mappo ippo_tuned mappo_act_adv)
CART_ALGOS=(MAPPO IPPO MAPPO)
CART_AGENTS=("" skrl_ippo_tuned_cfg_entry_point "")
CART_CKPTS=(
  "$ISAACLAB_ROOT/logs/skrl/cart_double_pendulum_direct/2026-06-22_22-02-46_mappo_torch/checkpoints/best_agent.pt"
  "$ISAACLAB_ROOT/logs/skrl/cart_double_pendulum_direct/2026-06-27_20-11-31_ippo_tuned_torch/checkpoints/best_agent.pt"
  "$ISAACLAB_ROOT/logs/skrl/cart_double_pendulum_direct/2026-07-08_02-45-17_mappo_torch/checkpoints/best_agent.pt"
)
CART_MAPPO_CKPT="${CART_CKPTS[0]}"

SHADOW_TASK="Isaac-Shadow-Hand-Over-Direct-v0"
SHADOW_TAGS=(mappo ippo mappo_act_adv)
SHADOW_ALGOS=(MAPPO IPPO MAPPO)
SHADOW_AGENTS=(skrl_mappo_convergence_cfg_entry_point "" skrl_mappo_convergence_cfg_entry_point)
SHADOW_CKPTS=(
  "$ISAACLAB_ROOT/logs/skrl/shadow_hand_over/2026-06-22_22-34-50_mappo_convergence_torch/checkpoints/best_agent.pt"
  "$ISAACLAB_ROOT/logs/skrl/shadow_hand_over/2026-06-27_17-45-33_ippo_torch/checkpoints/best_agent.pt"
  "$ISAACLAB_ROOT/logs/skrl/shadow_hand_over/2026-07-13_01-20-41_mappo_torch/checkpoints/best_agent.pt"
)
SHADOW_MAPPO_CKPT="${SHADOW_CKPTS[0]}"
SHADOW_MAPPO_AGENT="${SHADOW_AGENTS[0]}"

CART_DIFFUSION_MODEL="$ISAACLAB_ROOT/results/diffusion_model_cart.pt"
SHADOW_DIFFUSION_MODEL="$ISAACLAB_ROOT/results/diffusion_model_shadow.pt"
DIFFUSION_T_START_LIST="20 40"

DELAY_SWEEP_DIR="$ISAACLAB_ROOT/results/delay_sweep"
mkdir -p "$DELAY_SWEEP_DIR/cart" "$DELAY_SWEEP_DIR/shadow"

run_cart() {
  local tag=$1 algo=$2 agent=$3 ckpt=$4
  local agent_args=()
  [[ -n "$agent" ]] && agent_args=(--agent "$agent")
  echo ""; echo "[Cart] tag=$tag algorithm=$algo"
  conda run -n isaaclab bash "$ISAACLAB_ROOT/isaaclab.sh" -p "$SCRIPT" \
    --task "$CART_TASK" --algorithm "$algo" "${agent_args[@]}" \
    --checkpoint "$ckpt" --attack_kind delay --delay_ks $DELAY_K_LIST \
    --num_envs $NUM_ENVS --num_episodes $NUM_EPISODES --headless \
    --out "$ISAACLAB_ROOT/results/${tag}_delay_sweep_cart.txt"
}

run_shadow() {
  local tag=$1 algo=$2 agent=$3 ckpt=$4
  local agent_args=()
  [[ -n "$agent" ]] && agent_args=(--agent "$agent")
  echo ""; echo "[Shadow] tag=$tag algorithm=$algo"
  conda run -n isaaclab bash "$ISAACLAB_ROOT/isaaclab.sh" -p "$SCRIPT" \
    --task "$SHADOW_TASK" --algorithm "$algo" "${agent_args[@]}" \
    --checkpoint "$ckpt" --attack_kind delay --delay_ks $DELAY_K_LIST \
    --num_envs $NUM_ENVS --num_episodes $NUM_EPISODES \
    --success_threshold 0.1 --reward_scale 2.0 --dist_reward_scale 20.0 \
    --headless \
    --out "$ISAACLAB_ROOT/results/${tag}_delay_sweep_shadow.txt"
}

for i in "${!CART_TAGS[@]}"; do
  run_cart "${CART_TAGS[$i]}" "${CART_ALGOS[$i]}" "${CART_AGENTS[$i]}" "${CART_CKPTS[$i]}"
done
for i in "${!SHADOW_TAGS[@]}"; do
  run_shadow "${SHADOW_TAGS[$i]}" "${SHADOW_ALGOS[$i]}" "${SHADOW_AGENTS[$i]}" "${SHADOW_CKPTS[$i]}"
done

# --- Diffusion denoiser, evaluated against all three checkpoints (denoiser itself was
# trained only on vanilla-MAPPO trajectories; the ippo/mappo_act_adv runs test whether
# it generalizes to other policies' action distributions) ---
run_cart_diffusion() {
  local tag=$1 algo=$2 agent=$3 ckpt=$4
  local agent_args=()
  [[ -n "$agent" ]] && agent_args=(--agent "$agent")
  echo ""; echo "[Cart] tag=${tag}_diffusion algorithm=$algo (denoiser sweep)"
  conda run -n isaaclab bash "$ISAACLAB_ROOT/isaaclab.sh" -p "$DIFFUSION_SCRIPT" \
    --task "$CART_TASK" --algorithm "$algo" "${agent_args[@]}" \
    --checkpoint "$ckpt" --diffusion_model "$CART_DIFFUSION_MODEL" \
    --attack_kind delay --delay_ks $DELAY_K_LIST --t_start_list $DIFFUSION_T_START_LIST \
    --num_envs $NUM_ENVS --num_episodes $NUM_EPISODES --headless \
    --out "$DELAY_SWEEP_DIR/cart/${tag}_diffusion_delay_sweep_cart.txt"
}

run_shadow_diffusion() {
  local tag=$1 algo=$2 agent=$3 ckpt=$4
  local agent_args=()
  [[ -n "$agent" ]] && agent_args=(--agent "$agent")
  echo ""; echo "[Shadow] tag=${tag}_diffusion algorithm=$algo (denoiser sweep)"
  conda run -n isaaclab bash "$ISAACLAB_ROOT/isaaclab.sh" -p "$DIFFUSION_SCRIPT" \
    --task "$SHADOW_TASK" --algorithm "$algo" "${agent_args[@]}" \
    --checkpoint "$ckpt" --diffusion_model "$SHADOW_DIFFUSION_MODEL" \
    --attack_kind delay --delay_ks $DELAY_K_LIST --t_start_list $DIFFUSION_T_START_LIST \
    --num_envs $NUM_ENVS --num_episodes $NUM_EPISODES \
    --success_threshold 0.1 --reward_scale 2.0 --dist_reward_scale 20.0 \
    --headless \
    --out "$DELAY_SWEEP_DIR/shadow/${tag}_diffusion_delay_sweep_shadow.txt"
}

for i in "${!CART_TAGS[@]}"; do
  run_cart_diffusion "${CART_TAGS[$i]}" "${CART_ALGOS[$i]}" "${CART_AGENTS[$i]}" "${CART_CKPTS[$i]}"
done
for i in "${!SHADOW_TAGS[@]}"; do
  run_shadow_diffusion "${SHADOW_TAGS[$i]}" "${SHADOW_ALGOS[$i]}" "${SHADOW_AGENTS[$i]}" "${SHADOW_CKPTS[$i]}"
done

# --- Combined summary ---
echo ""
echo "============================================================"
echo "  RESULTS"
echo "============================================================"
for i in "${!CART_TAGS[@]}"; do
  echo ""; echo "=== Cart Double Pendulum: ${CART_TAGS[$i]} ==="
  cat "$ISAACLAB_ROOT/results/${CART_TAGS[$i]}_delay_sweep_cart.txt"
  echo ""; echo "=== Cart Double Pendulum: ${CART_TAGS[$i]}_diffusion ==="
  cat "$DELAY_SWEEP_DIR/cart/${CART_TAGS[$i]}_diffusion_delay_sweep_cart.txt"
done

for i in "${!SHADOW_TAGS[@]}"; do
  echo ""; echo "=== Shadow Hand Over: ${SHADOW_TAGS[$i]} ==="
  cat "$ISAACLAB_ROOT/results/${SHADOW_TAGS[$i]}_delay_sweep_shadow.txt"
  echo ""; echo "=== Shadow Hand Over: ${SHADOW_TAGS[$i]}_diffusion ==="
  cat "$DELAY_SWEEP_DIR/shadow/${SHADOW_TAGS[$i]}_diffusion_delay_sweep_shadow.txt"
done
