#!/usr/bin/env bash
set -euo pipefail
# Run SFT LoRA ablation experiments from one base YAML.
#
# Experiments:
#   1. rank=8,  target_modules=q_proj+v_proj
#   2. rank=16, target_modules=q_proj+v_proj
#   3. rank=32, target_modules=q_proj+v_proj
#   4. rank=16, target_modules=q_proj+k_proj+v_proj+o_proj
#   5. rank=16, target_modules=all main Qwen linear layers
#
# Usage:
#   bash SFT_DPO/train/run_sft_ablation_experiments.sh
#
# Optional:
#   BASE_CONFIG=/path/to/base.yaml bash SFT_DPO/train/run_sft_ablation_experiments.sh
#   DRY_RUN=1 bash SFT_DPO/train/run_sft_ablation_experiments.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BASE_CONFIG="${BASE_CONFIG:-${SCRIPT_DIR}/configs/trl_qwen_customer_sft.yaml}"
GENERATED_CONFIG_DIR="${GENERATED_CONFIG_DIR:-${SCRIPT_DIR}/configs/experiments}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/sft_ablation}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${SCRIPT_DIR}/train_trl_sft.py}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${GENERATED_CONFIG_DIR}" "${OUTPUT_ROOT}"

make_config() {
  local run_name="$1"
  local rank="$2"
  local alpha="$3"
  local modules_csv="$4"
  local config_path="${GENERATED_CONFIG_DIR}/${run_name}.yaml"
  local output_dir="${OUTPUT_ROOT}/${run_name}/checkpoint"
  local artifact_dir="${output_dir}/artifacts"

  "${PYTHON_BIN}" - "${BASE_CONFIG}" "${config_path}" "${run_name}" "${rank}" "${alpha}" "${modules_csv}" "${output_dir}" "${artifact_dir}" <<'PY'
import sys
from pathlib import Path

import yaml

base_config, config_path, run_name, rank, alpha, modules_csv, output_dir, artifact_dir = sys.argv[1:]

with open(base_config, "r", encoding="utf-8") as file_obj:
    config = yaml.safe_load(file_obj)

target_modules = [item.strip() for item in modules_csv.split(",") if item.strip()]

config["lora"]["rank"] = int(rank)
config["lora"]["alpha"] = int(alpha)
config["lora"]["target_modules"] = target_modules

config["training"]["resume_from_checkpoint"] = None
config["output"]["output_dir"] = str(Path(output_dir).resolve())
config["output"]["artifact_dir"] = str(Path(artifact_dir).resolve())

Path(config_path).parent.mkdir(parents=True, exist_ok=True)
with open(config_path, "w", encoding="utf-8") as file_obj:
    yaml.safe_dump(config, file_obj, allow_unicode=True, sort_keys=False)

print(config_path)
PY
}

run_experiment() {
  local run_name="$1"
  local rank="$2"
  local alpha="$3"
  local modules_csv="$4"
  local config_path

  config_path="$(make_config "${run_name}" "${rank}" "${alpha}" "${modules_csv}")"

  echo
  echo "============================================================"
  echo "Run: ${run_name}"
  echo "Config: ${config_path}"
  echo "rank=${rank}, alpha=${alpha}, target_modules=${modules_csv}"
  echo "============================================================"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY_RUN] ${PYTHON_BIN} ${TRAIN_SCRIPT} --config ${config_path}"
  else
    "${PYTHON_BIN}" "${TRAIN_SCRIPT}" --config "${config_path}"
  fi
}

EXPERIMENTS=(
  # Rank ablation with q_proj+v_proj.
  # "rank8_qv|8|16|q_proj,v_proj"
  "rank16_qv|16|32|q_proj,v_proj"
  "rank32_qv|32|64|q_proj,v_proj"
  # Target-module ablation at rank=16.
  "rank16_qvko|16|32|q_proj,k_proj,v_proj,o_proj"
  "rank16_all_linear|16|32|q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
)

for experiment in "${EXPERIMENTS[@]}"; do
  IFS="|" read -r run_name rank alpha modules_csv <<< "${experiment}"
  run_experiment "${run_name}" "${rank}" "${alpha}" "${modules_csv}"
done

echo
echo "All SFT ablation experiments finished."
echo "Generated configs: ${GENERATED_CONFIG_DIR}"
echo "Outputs: ${OUTPUT_ROOT}"
