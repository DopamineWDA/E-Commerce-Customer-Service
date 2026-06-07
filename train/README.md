# TRL SFT for JD Customer Service

This directory rewrites the LLaMA-Factory LoRA SFT flow into a standalone TRL pipeline.

## Files

- `configs/trl_qwen_customer_sft.yaml`: training config
- `configs/deepspeed/ds_zero2.json`: optional DeepSpeed ZeRO-2 config
- `train_trl_sft.py`: main SFT training script
- `train_trl_dpo.py`: DPO preference-alignment script using TRL `DPOTrainer`
- `plot_loss.py`: re-plot `trainer_state.json` into a PNG curve
- `infer_trl_sft.py`: generate inference samples from the saved LoRA adapter
- `evaluate_inference_judge.py`: score inference samples with a qwen-plus judge
- `sample_prompts.json`: example prompts for manual testing
- `requirements-train.txt`: Python dependencies

## Why this matches the request

- Trainer: `trl.SFTTrainer`
- LoRA: `rank=16`, `target_modules=["q_proj", "v_proj"]`
- Epochs: default `3.0`
- Loss curve: exported to `artifacts/loss_curve.png`
- Inference samples: exported to `artifacts/inference_samples.json`
- Data format: directly consumes your ShareGPT-style `conversations`

## Install

```powershell
pip install -r Main/train/requirements-train.txt
```

## Train

```powershell
python Main/train/train_trl_sft.py --config Main/train/configs/trl_qwen_customer_sft.yaml
```

## Train DPO

The DPO flow uses `/home/txs/work/zyp/SFT_DPO/data_dpo/dpo_pairs.jsonl`
and starts from the SFT LoRA adapter at
`/home/txs/work/zyp/SFT_DPO/train/outputs/sft_ablation/rank16_qvko/checkpoint`.
The default DPO beta is recorded explicitly as `dpo.beta: 0.1`.

```powershell
python SFT_DPO/train/train_trl_dpo.py --config SFT_DPO/train/configs/trl_qwen_customer_dpo.yaml
```

Outputs are written to:

- `SFT_DPO/train/outputs/dpo/rank16_qvko_beta0.1/checkpoint`
- `SFT_DPO/train/outputs/dpo/rank16_qvko_beta0.1/checkpoint/artifacts/loss_curve.png`
- `SFT_DPO/train/outputs/dpo/rank16_qvko_beta0.1/checkpoint/artifacts/inference_samples.json`
- `SFT_DPO/train/outputs/dpo/rank16_qvko_beta0.1/checkpoint/artifacts/run_summary.json`

## Enable DeepSpeed ZeRO-2

Edit `Main/train/configs/trl_qwen_customer_sft.yaml`:

```yaml
distributed:
  deepspeed: deepspeed/ds_zero2.json
```

Then launch with a distributed launcher. A common example is:

```powershell
torchrun --nproc_per_node=2 Main/train/train_trl_sft.py --config Main/train/configs/trl_qwen_customer_sft.yaml
```

Notes:

- The training script now passes `deepspeed=...` into `SFTConfig`.
- Relative DeepSpeed paths are resolved from the YAML directory automatically.
- When DeepSpeed is enabled, the script avoids `device_map="auto"` and leaves device placement to the launcher.
- ZeRO-2 is most useful for multi-GPU training rather than simple single-GPU runs.
- Whether it can actually run on your machine still depends on your local DeepSpeed installation and platform support.

## Re-plot loss

```powershell
python Main/train/plot_loss.py `
  --trainer-state Main/train/outputs/qwen2_5_1_5b_jd_lora_sft/trainer_state.json `
  --output Main/train/outputs/qwen2_5_1_5b_jd_lora_sft/artifacts/loss_curve.png
```

## Run inference

```powershell
python Main/train/infer_trl_sft.py --config Main/train/configs/trl_qwen_customer_sft.yaml
```

## Judge inference samples

The judge script reads `artifacts/inference_samples.json`, calls qwen-plus, and
exports per-sample scores plus aggregate metrics:

- `Accuracy`: mean of 0/1 answer accuracy
- `Auto Resolve`: mean of 0/1 automatic-resolution score
- `Closure Rate`: mean of 0/1 closure score
- `CSAT`: mean satisfaction score from 1 to 5

```powershell
$env:OPENAI_API_KEY = "your_api_key"
python Main/train/evaluate_inference_judge.py `
  --input Main/train/outputs/qwen3-8b_lora_sft_5.7_1/checkpoint/artifacts/inference_samples.json `
  --model qwen-plus
```

Outputs are written next to the input file by default:

- `judge_results.jsonl`
- `judge_summary.json`
- `judge_summary.csv`

By default the script uses `OPENAI_BASE_URL` if set, otherwise
`https://www.dmxapi.com/v1`, matching the data augmentation script.

## Reference mapping from LLaMA-Factory

- Reference train config: `LlamaFactory-ref/examples/train_lora/qwen3_lora_sft.yaml`
- Reference infer config: `LlamaFactory-ref/examples/inference/qwen3_lora_sft.yaml`
- Your dataset format: `Main/data/dataset_info.json`

## Notes

- The YAML uses `Qwen/Qwen2.5-1.5B-Instruct` by default because it is much more practical on a single RTX 3060.
- If you want to stay closer to the LLaMA-Factory example, replace `model.model_name_or_path` with `Qwen/Qwen3-4B-Instruct-2507`, but VRAM pressure will be much higher.
