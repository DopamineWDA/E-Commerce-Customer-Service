#!/usr/bin/env bash
set -euo pipefail

# Run DPO beta ablation experiments on the same preference data.
#
# Experiments:
#   beta = 0.05 / 0.1 / 0.3
#
# Flow for each beta:
#   1. Generate a DPO config from configs/trl_qwen_customer_dpo.yaml.
#   2. Train with /home/txs/work/zyp/SFT_DPO/data_dpo/dpo_pairs.jsonl.
#   3. Pick the best checkpoint recorded by Trainer.
#   4. Run infer_trl_sft.py with the best checkpoint and save artifacts/infer.jsonl.
#   5. Evaluate artifacts/infer.jsonl with evaluate_inference_judge.py.
#   6. Write an aggregate experiment summary.
#
# Usage:
#   bash SFT_DPO/train/run_dpo_beta_ablation_experiments.sh
#
# Useful overrides:
#   DRY_RUN=1 bash SFT_DPO/train/run_dpo_beta_ablation_experiments.sh
#   TORCHRUN_NPROC=2 bash SFT_DPO/train/run_dpo_beta_ablation_experiments.sh
#   SKIP_TRAIN=1 bash SFT_DPO/train/run_dpo_beta_ablation_experiments.sh
#   SKIP_INFER=1 bash SFT_DPO/train/run_dpo_beta_ablation_experiments.sh
#   SKIP_JUDGE=1 bash SFT_DPO/train/run_dpo_beta_ablation_experiments.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BASE_DPO_CONFIG="${BASE_DPO_CONFIG:-${SCRIPT_DIR}/configs/trl_qwen_customer_dpo.yaml}"
BASE_INFER_CONFIG="${BASE_INFER_CONFIG:-${SCRIPT_DIR}/configs/trl_qwen_customer_sft.yaml}"
DPO_TRAIN_FILE="${DPO_TRAIN_FILE:-/home/txs/work/zyp/SFT_DPO/data_dpo/dpo_pairs.jsonl}"
INFER_VAL_FILE="${INFER_VAL_FILE:-/home/txs/work/zyp/SFT_DPO/data_sft/val.jsonl}"

GENERATED_CONFIG_DIR="${GENERATED_CONFIG_DIR:-${SCRIPT_DIR}/configs/experiments/dpo_beta_ablation}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/dpo_beta_ablation/dpo_pairs}"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-${SCRIPT_DIR}/train_trl_dpo.py}"
INFER_SCRIPT="${INFER_SCRIPT:-${SCRIPT_DIR}/infer_trl_sft.py}"
JUDGE_SCRIPT="${JUDGE_SCRIPT:-${SCRIPT_DIR}/evaluate_inference_judge.py}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
TORCHRUN_NPROC="${TORCHRUN_NPROC:-2}"

DRY_RUN="${DRY_RUN:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_INFER="${SKIP_INFER:-0}"
SKIP_JUDGE="${SKIP_JUDGE:-0}"
OVERWRITE_JUDGE="${OVERWRITE_JUDGE:-1}"

JUDGE_MODEL="${JUDGE_MODEL:-qwen-plus}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-4}"
JUDGE_LIMIT="${JUDGE_LIMIT:-}"
INFER_NUM_EXAMPLES="${INFER_NUM_EXAMPLES:-}"

BETAS=(${BETAS:-0.05 0.1 0.3})

mkdir -p "${GENERATED_CONFIG_DIR}" "${OUTPUT_ROOT}"

echo "DPO beta list: ${BETAS[*]}"
echo "DPO train file: ${DPO_TRAIN_FILE}"
echo "Infer val file: ${INFER_VAL_FILE}"

run_cmd() {
  echo "+ $*"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

run_cmd_logged() {
  local log_path="$1"
  shift

  mkdir -p "$(dirname "${log_path}")"
  echo "+ $*"
  echo "+ $*" > "${log_path}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@" 2>&1 | tee -a "${log_path}"
  fi
}

make_dpo_config() {
  local beta="$1"
  local run_name="$2"
  local config_path="${GENERATED_CONFIG_DIR}/${run_name}.yaml"
  local output_dir="${OUTPUT_ROOT}/${run_name}/checkpoint"
  local artifact_dir="${output_dir}/artifacts"

  "${PYTHON_BIN}" - "${BASE_DPO_CONFIG}" "${config_path}" "${DPO_TRAIN_FILE}" "${beta}" "${output_dir}" "${artifact_dir}" <<'PY'
import sys
from pathlib import Path

import yaml

base_config, config_path, train_file, beta, output_dir, artifact_dir = sys.argv[1:]

with open(base_config, "r", encoding="utf-8") as file_obj:
    config = yaml.safe_load(file_obj)

config["dataset"]["train_file"] = str(Path(train_file).resolve())
config["dataset"]["val_file"] = None
config["dpo"]["beta"] = float(beta)
deepspeed_config = config.get("distributed", {}).get("deepspeed")
if deepspeed_config:
    deepspeed_path = Path(deepspeed_config)
    if not deepspeed_path.is_absolute():
        deepspeed_path = (Path(base_config).resolve().parent / deepspeed_path).resolve()
    config["distributed"]["deepspeed"] = str(deepspeed_path)
config["training"]["resume_from_checkpoint"] = None
config["training"]["overwrite_output_dir"] = False
config["output"]["output_dir"] = str(Path(output_dir).resolve())
config["output"]["artifact_dir"] = str(Path(artifact_dir).resolve())

Path(config_path).parent.mkdir(parents=True, exist_ok=True)
with open(config_path, "w", encoding="utf-8") as file_obj:
    yaml.safe_dump(config, file_obj, allow_unicode=True, sort_keys=False)

print(config_path)
PY
}

make_infer_config() {
  local run_name="$1"
  local config_path="${GENERATED_CONFIG_DIR}/${run_name}_infer.yaml"
  local output_dir="${OUTPUT_ROOT}/${run_name}/checkpoint"
  local artifact_dir="${output_dir}/artifacts"

  "${PYTHON_BIN}" - "${BASE_INFER_CONFIG}" "${config_path}" "${INFER_VAL_FILE}" "${output_dir}" "${artifact_dir}" <<'PY'
import sys
from pathlib import Path

import yaml

base_config, config_path, val_file, output_dir, artifact_dir = sys.argv[1:]

with open(base_config, "r", encoding="utf-8") as file_obj:
    config = yaml.safe_load(file_obj)

config["dataset"]["val_file"] = str(Path(val_file).resolve())
config["output"]["output_dir"] = str(Path(output_dir).resolve())
config["output"]["artifact_dir"] = str(Path(artifact_dir).resolve())

Path(config_path).parent.mkdir(parents=True, exist_ok=True)
with open(config_path, "w", encoding="utf-8") as file_obj:
    yaml.safe_dump(config, file_obj, allow_unicode=True, sort_keys=False)

print(config_path)
PY
}

best_checkpoint_for_run() {
  local output_dir="$1"
  local run_summary="${output_dir}/artifacts/run_summary.json"
  local trainer_state="${output_dir}/trainer_state.json"

  "${PYTHON_BIN}" - "${run_summary}" "${trainer_state}" "${output_dir}" <<'PY'
import json
import sys
from pathlib import Path

run_summary, trainer_state, output_dir = [Path(item) for item in sys.argv[1:]]

best = None
for path in (run_summary, trainer_state):
    if not path.exists():
        continue
    with path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    best = payload.get("best_model_checkpoint")
    if best:
        break

if not best:
    best = str(output_dir)

best_path = Path(best)
if not best_path.exists() and not best_path.is_absolute():
    best_path = (output_dir / best_path).resolve()

if not best_path.exists():
    raise SystemExit(f"Best checkpoint does not exist: {best}")

print(best_path)
PY
}

write_summary() {
  local summary_csv="${OUTPUT_ROOT}/beta_ablation_summary.csv"
  local summary_md="${OUTPUT_ROOT}/BETA_ABLATION_SUMMARY.md"

  "${PYTHON_BIN}" - "${OUTPUT_ROOT}" "${summary_csv}" "${summary_md}" <<'PY'
import csv
import json
import sys
from pathlib import Path

output_root, summary_csv, summary_md = [Path(item) for item in sys.argv[1:]]

rows = []
for run_dir in sorted(output_root.glob("rank16_qvko_beta*/checkpoint")):
    artifact_dir = run_dir / "artifacts"
    run_summary_path = artifact_dir / "run_summary.json"
    judge_summary_path = artifact_dir / "judge_summary.json"

    run_summary = {}
    judge_summary = {}
    if run_summary_path.exists():
        run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    if judge_summary_path.exists():
        judge_summary = json.loads(judge_summary_path.read_text(encoding="utf-8"))

    rows.append(
        {
            "run_name": run_dir.parent.name,
            "beta": run_summary.get("dpo_beta", run_dir.parent.name.replace("rank16_qvko_beta", "")),
            "train_samples": run_summary.get("train_samples"),
            "eval_samples": run_summary.get("eval_samples"),
            "global_step": run_summary.get("global_step"),
            "best_metric": run_summary.get("best_metric"),
            "best_model_checkpoint": run_summary.get("best_model_checkpoint"),
            "judge_samples": judge_summary.get("num_samples"),
            "accuracy_percent": judge_summary.get("accuracy_percent"),
            "auto_resolve_percent": judge_summary.get("auto_resolve_percent"),
            "closure_rate_percent": judge_summary.get("closure_rate_percent"),
            "csat_1_to_5": judge_summary.get("csat_1_to_5"),
            "infer_path": str(artifact_dir / "infer.jsonl"),
            "judge_summary": str(judge_summary_path),
        }
    )

fieldnames = [
    "run_name",
    "beta",
    "train_samples",
    "eval_samples",
    "global_step",
    "best_metric",
    "best_model_checkpoint",
    "judge_samples",
    "accuracy_percent",
    "auto_resolve_percent",
    "closure_rate_percent",
    "csat_1_to_5",
    "infer_path",
    "judge_summary",
]

summary_csv.parent.mkdir(parents=True, exist_ok=True)
with summary_csv.open("w", encoding="utf-8", newline="") as file_obj:
    writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

lines = [
    "# DPO Beta Ablation Summary",
    "",
    f"- Output root: `{output_root}`",
    f"- Summary CSV: `{summary_csv}`",
    "",
    "| beta | best_metric | judge_samples | accuracy | auto_resolve | closure | csat |",
    "|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    def cell(key):
        value = row.get(key)
        return "" if value is None else str(value)

    lines.append(
        "| "
        + " | ".join(
            [
                cell("beta"),
                cell("best_metric"),
                cell("judge_samples"),
                cell("accuracy_percent"),
                cell("auto_resolve_percent"),
                cell("closure_rate_percent"),
                cell("csat_1_to_5"),
            ]
        )
        + " |"
    )

summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary_csv)
print(summary_md)
PY
}

for beta in "${BETAS[@]}"; do
  beta_tag="${beta//./p}"
  run_name="rank16_qvko_beta${beta_tag}"
  config_path="$(make_dpo_config "${beta}" "${run_name}")"
  output_dir="${OUTPUT_ROOT}/${run_name}/checkpoint"
  artifact_dir="${output_dir}/artifacts"
  infer_output="${artifact_dir}/infer.jsonl"

  echo
  echo "============================================================"
  echo "Run: ${run_name}"
  echo "Beta: ${beta}"
  echo "Train file: ${DPO_TRAIN_FILE}"
  echo "Config: ${config_path}"
  echo "Output: ${output_dir}"
  echo "============================================================"

  if [[ "${SKIP_TRAIN}" != "1" ]]; then
    run_cmd_logged "${artifact_dir}/train.log" \
      env PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 TORCH_DISTRIBUTED_DEBUG=DETAIL \
      "${TORCHRUN_BIN}" --nproc_per_node="${TORCHRUN_NPROC}" "${TRAIN_SCRIPT}" --config "${config_path}"
  fi

  best_checkpoint="$(best_checkpoint_for_run "${output_dir}")"
  echo "Best checkpoint: ${best_checkpoint}"

  if [[ "${SKIP_INFER}" != "1" ]]; then
    infer_config_path="$(make_infer_config "${run_name}")"
    infer_args=(
      "${PYTHON_BIN}" "${INFER_SCRIPT}"
      --config "${infer_config_path}"
      --adapter-path "${best_checkpoint}"
      --output "${infer_output}"
    )
    if [[ -n "${INFER_NUM_EXAMPLES}" ]]; then
      infer_args+=(--num-examples "${INFER_NUM_EXAMPLES}")
    fi
    run_cmd_logged "${artifact_dir}/infer.log" "${infer_args[@]}"
  fi

  if [[ "${SKIP_JUDGE}" != "1" ]]; then
    judge_args=(
      "${PYTHON_BIN}" "${JUDGE_SCRIPT}"
      --input "${infer_output}"
      --output-dir "${artifact_dir}"
      --model "${JUDGE_MODEL}"
      --concurrency "${JUDGE_CONCURRENCY}"
    )
    if [[ "${OVERWRITE_JUDGE}" == "1" ]]; then
      judge_args+=(--overwrite)
    fi
    if [[ -n "${JUDGE_LIMIT}" ]]; then
      judge_args+=(--limit "${JUDGE_LIMIT}")
    fi
    run_cmd_logged "${artifact_dir}/judge.log" "${judge_args[@]}"
  fi
done

write_summary

echo
echo "All DPO beta ablation experiments finished."
echo "Generated configs: ${GENERATED_CONFIG_DIR}"
echo "Outputs: ${OUTPUT_ROOT}"
