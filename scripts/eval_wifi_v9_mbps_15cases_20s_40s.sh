#!/usr/bin/env bash
set -euo pipefail

# Run from the git repository root:
#   bash scripts/eval_wifi_v9_mbps_15cases_20s_40s.sh
#
# Execution order:
#   1. RL 20 sec eval
#   2. RL 40 sec eval
#   3. BEB 20 sec eval
#   4. BEB 40 sec eval
#   5. BEB vs RL 20 sec plots
#   6. BEB vs RL 40 sec plots

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

MODEL_DIR="/home/se/repos/mappo/model/WiFi_v9/mappo/wifi_v9_train_airtime50ms_m15m25_s3s5_parallel_vec4_d2lt_mldsucc1_sld07_10_ntop1_cidle03_1600k_lr1e4_ent5e3_seed1"
RESULT_BASE="/home/se/repos/scripts/eval_results/WiFi_v9"
WANDB_ENTITY="${WANDB_ENTITY:-nhw1124-kumoh-national-institute-of-technology}"
EVAL_EPISODES="${EVAL_EPISODES:-4}"
SEED="${SEED:-1}"

TAG="parallel_vec4_d2lt_mldsucc1_sld07_10_ntop1_cidle03"

COMMON_ARGS=(
  --env_name WiFi_v9
  --algorithm_name mappo
  --max_mld 30
  --max_sld 10
  --round_length 500
  --mu_min 0.01
  --mu_max 0.12
  --eta 0.2
  --zeta 0.2
  --c_idle 0.3
  --collision_penalty 1.0
  --non_top_tx_penalty 1.0
  --theta_scale 1.0
  --sld_target_low_scale 0.7
  --sld_target_high_scale 1.0
  --sld_target_bonus 0.5
  --mld_success_reward 1.0
  --eval_episodes "${EVAL_EPISODES}"
  --slot_time_sec 9e-6
  --seed "${SEED}"
  --wandb_entity "${WANDB_ENTITY}"
)

CASES=(
  "10 2"
  "15 2"
  "20 2"
  "25 2"
  "30 2"
  "10 4"
  "15 4"
  "20 4"
  "25 4"
  "30 4"
  "10 6"
  "15 6"
  "20 6"
  "25 6"
  "30 6"
)

experiment_name() {
  local policy="$1"
  local duration="$2"
  local mld="$3"
  local sld="$4"

  printf "wifi_v9_%s_mbps_15case_%s_%ss_m%s_s%s_airtime_cw16_ep%s" \
    "${policy}" "${TAG}" "${duration}" "${mld}" "${sld}" "${EVAL_EPISODES}"
}

run_rl_duration() {
  local duration="$1"

  echo "========== RL ${duration}s evaluation =========="
  for item in "${CASES[@]}"; do
    read -r mld sld <<< "${item}"
    exp_name="$(experiment_name rl "${duration}" "${mld}" "${sld}")"

    echo "[RL ${duration}s] m${mld}_s${sld}"
    python -m onpolicy.scripts.eval.eval_wifi_v9_rl_mbps \
      "${COMMON_ARGS[@]}" \
      --experiment_name "${exp_name}" \
      --num_mld "${mld}" \
      --num_sld "${sld}" \
      --eval_duration_sec "${duration}" \
      --debug_prob_steps 0 \
      --stochastic \
      --wandb_project "WiFi_v9_rl_eval_mbps_${duration}s_15case" \
      --wandb_group "wifi_v9_15case_${TAG}_${duration}s" \
      --wandb_run_name "rl_15case_${TAG}_${duration}s_m${mld}_s${sld}_ep${EVAL_EPISODES}" \
      --model_dir "${MODEL_DIR}"
  done
}

run_beb_duration() {
  local duration="$1"

  echo "========== BEB ${duration}s evaluation =========="
  for item in "${CASES[@]}"; do
    read -r mld sld <<< "${item}"
    exp_name="$(experiment_name beb "${duration}" "${mld}" "${sld}")"

    echo "[BEB ${duration}s] m${mld}_s${sld}"
    python -m onpolicy.scripts.eval.eval_wifi_v9_beb_mbps \
      "${COMMON_ARGS[@]}" \
      --experiment_name "${exp_name}" \
      --num_mld "${mld}" \
      --num_sld "${sld}" \
      --eval_duration_sec "${duration}" \
      --wandb_project "WiFi_v9_beb_eval_mbps_${duration}s_15case" \
      --wandb_group "wifi_v9_15case_${TAG}_${duration}s" \
      --wandb_run_name "beb_15case_${TAG}_${duration}s_m${mld}_s${sld}_ep${EVAL_EPISODES}"
  done
}

compare_duration() {
  local duration="$1"
  local output_dir="${RESULT_BASE}/wifi_v9_mbps_15case_compare_${duration}s_nolabel"
  local compare_args=()

  echo "========== BEB vs RL ${duration}s comparison =========="
  for item in "${CASES[@]}"; do
    read -r mld sld <<< "${item}"
    label="m${mld}_s${sld}"
    beb_exp="$(experiment_name beb "${duration}" "${mld}" "${sld}")"
    rl_exp="$(experiment_name rl "${duration}" "${mld}" "${sld}")"

    compare_args+=(
      --case "${label}|${RESULT_BASE}/wifi_v9_beb_mbps/${beb_exp}/beb_mbps_summary.json|${RESULT_BASE}/wifi_v9_rl_mbps/${rl_exp}/rl_mbps_summary.json"
    )
  done

  python -m onpolicy.scripts.eval.compare_wifi_mbps_cases \
    "${compare_args[@]}" \
    --output_dir "${output_dir}" \
    --output_name "wifi_v9_beb_vs_rl_${duration}s_15case_nolabel.png" \
    --title "WIFI v9 BEB vs RL Mbps Comparison (${duration}s)" \
    --hide_bar_labels \
    --wandb_project "WiFi_v9_mbps_compare_${duration}s_15case" \
    --wandb_group "wifi_v9_15case_${TAG}_${duration}s_nolabel" \
    --wandb_run_name "beb_vs_rl_15case_${TAG}_${duration}s_nolabel" \
    --wandb_entity "${WANDB_ENTITY}"
}

run_rl_duration 20
run_rl_duration 40
run_beb_duration 20
run_beb_duration 40
compare_duration 20
compare_duration 40
