#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Use TRL's DPOTrainer to align the SFT LoRA adapter with preference pairs."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import yaml
from datasets import DatasetDict, load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback


def ensure_utf8_mode() -> None:
    if not sys.flags.utf8_mode:
        os.environ["PYTHONUTF8"] = "1"
        os.execv(sys.executable, [sys.executable, *sys.argv])


def prefer_local_trl_ref() -> None:
    """Import TRL from the local trl-ref checkout when it is present."""
    project_root = Path(__file__).resolve().parents[2]
    trl_ref = project_root / "trl-ref"
    if trl_ref.is_dir() and str(trl_ref) not in sys.path:
        sys.path.insert(0, str(trl_ref))


def remove_local_trl_ref_from_import_state() -> None:
    project_root = Path(__file__).resolve().parents[2]
    trl_ref = str(project_root / "trl-ref")
    sys.path[:] = [item for item in sys.path if item != trl_ref]
    for module_name in list(sys.modules):
        if module_name == "trl" or module_name.startswith("trl."):
            del sys.modules[module_name]


def import_trl_objects():
    ensure_utf8_mode()
    prefer_local_trl_ref()
    try:
        from trl import DPOConfig, DPOTrainer
    except Exception as exc:
        remove_local_trl_ref_from_import_state()
        warnings.warn(
            "Importing TRL from local `trl-ref` failed, so the script is falling back to the installed `trl` package. "
            f"Original error: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        from trl import DPOConfig, DPOTrainer

    return DPOConfig, DPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TRL DPO from local preference pairs.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "configs" / "trl_qwen_customer_dpo.yaml"),
        help="Path to YAML config.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)
    config["__config_dir__"] = str(Path(path).resolve().parent)
    return config


def resolve_from_config_dir(config: dict[str, Any], maybe_path: str | None) -> str | None:
    if not maybe_path:
        return None
    candidate = Path(maybe_path)
    if candidate.is_absolute():
        return str(candidate)
    return str((Path(config["__config_dir__"]) / candidate).resolve())


def get_deepspeed_config_path(config: dict[str, Any]) -> str | None:
    return resolve_from_config_dir(config, config.get("distributed", {}).get("deepspeed"))


def is_deepspeed_enabled(config: dict[str, Any]) -> bool:
    return get_deepspeed_config_path(config) is not None


def validate_optional_deepspeed(config: dict[str, Any]) -> None:
    if not is_deepspeed_enabled(config):
        return
    try:
        importlib.metadata.version("deepspeed")
    except importlib.metadata.PackageNotFoundError as exc:
        ds_path = get_deepspeed_config_path(config)
        raise RuntimeError(
            "DeepSpeed is enabled in the YAML config, but the `deepspeed` package is not installed. "
            f"Configured file: {ds_path}. Install DeepSpeed first, or set `distributed.deepspeed:` to empty."
        ) from exc


def get_distributed_context() -> dict[str, int | bool]:
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return {
        "local_rank": local_rank,
        "rank": rank,
        "world_size": world_size,
        "is_distributed": world_size > 1,
        "is_main_process": rank == 0,
    }


def setup_distributed_device(distributed_ctx: dict[str, int | bool]) -> None:
    if not distributed_ctx["is_distributed"]:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed launch was detected, but CUDA is not available.")
    local_rank = int(distributed_ctx["local_rank"])
    if local_rank < 0:
        raise RuntimeError("Distributed launch was detected, but LOCAL_RANK is missing.")
    torch.cuda.set_device(local_rank)


def pick_precision() -> dict[str, bool | torch.dtype]:
    if not torch.cuda.is_available():
        return {"bf16": False, "fp16": False, "dtype": torch.float32}
    if torch.cuda.is_bf16_supported():
        return {"bf16": True, "fp16": False, "dtype": torch.bfloat16}
    return {"bf16": False, "fp16": True, "dtype": torch.float16}


def build_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # DPO preference batches are padded on the left in TRL's current trainer.
    tokenizer.padding_side = "left"
    return tokenizer


def _resolve_training_chat_template_path(model_name_or_path: str) -> Path | None:
    project_root = Path(__file__).resolve().parents[2]
    templates_dir = project_root / "trl-ref" / "trl" / "chat_templates"
    model_name = model_name_or_path.lower()
    if "qwen2.5" in model_name or "qwen2-5" in model_name:
        return templates_dir / "qwen2_5_training.jinja"
    if "qwen3.6" in model_name or "qwen3-6" in model_name:
        return templates_dir / "qwen3_6_training.jinja"
    if "qwen3" in model_name:
        return templates_dir / "qwen3_training.jinja"
    return None


def ensure_chat_template(tokenizer, model_name_or_path: str) -> None:
    if getattr(tokenizer, "chat_template", None):
        return
    template_path = _resolve_training_chat_template_path(model_name_or_path)
    if template_path is None or not template_path.is_file():
        raise RuntimeError(f"No chat template found for `{model_name_or_path}`.")
    tokenizer.chat_template = template_path.read_text(encoding="utf-8")


def build_datasets(config: dict[str, Any]) -> DatasetDict:
    dataset_cfg = config["dataset"]
    data_files = {"train": dataset_cfg["train_file"]}
    if dataset_cfg.get("val_file"):
        data_files["validation"] = dataset_cfg["val_file"]
        dataset = load_dataset("json", data_files=data_files)
    else:
        raw = load_dataset("json", data_files=data_files, split="train")
        split = raw.train_test_split(
            test_size=dataset_cfg.get("validation_split", 0.1),
            seed=dataset_cfg.get("split_seed", 42),
        )
        dataset = DatasetDict({"train": split["train"], "validation": split["test"]})

    required = {"prompt", "chosen", "rejected"}
    missing = required - set(dataset["train"].column_names)
    if missing:
        raise ValueError(f"DPO dataset is missing required columns: {sorted(missing)}")

    max_train_samples = dataset_cfg.get("max_train_samples")
    if max_train_samples:
        dataset["train"] = dataset["train"].select(range(min(max_train_samples, len(dataset["train"]))))

    max_eval_samples = dataset_cfg.get("max_eval_samples")
    if max_eval_samples:
        dataset["validation"] = dataset["validation"].select(range(min(max_eval_samples, len(dataset["validation"]))))

    return dataset


def build_policy_model(
    config: dict[str, Any],
    precision: dict[str, bool | torch.dtype],
    distributed_ctx,
    clone_ref_adapter: bool,
):
    model_cfg = config["model"]
    use_device_map = not distributed_ctx["is_distributed"] and not is_deepspeed_enabled(config)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_name_or_path"],
        trust_remote_code=True,
        torch_dtype=precision["dtype"],
        device_map="auto" if use_device_map and torch.cuda.is_available() else None,
    )
    if config["training"].get("gradient_checkpointing", True):
        base_model.config.use_cache = False

    adapter_path = model_cfg["sft_adapter_path"]
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
    if clone_ref_adapter:
        clone_peft_adapter(model, source_adapter="default", target_adapter="ref")
    model.set_adapter("default")
    return model


def clone_peft_adapter(model: PeftModel, source_adapter: str, target_adapter: str) -> None:
    """Copy the loaded SFT adapter so DPO can use it as the reference policy.

    Some TRL versions compute PEFT references by disabling adapters, which makes
    the reference the base model rather than the SFT model. For DPO after SFT we
    want pi_ref to be the frozen initial SFT adapter, so we create a second
    adapter with identical weights and pass it as `ref_adapter_name`.
    """
    if target_adapter in model.peft_config:
        return
    if source_adapter not in model.peft_config:
        raise ValueError(f"Source adapter `{source_adapter}` does not exist. Available: {list(model.peft_config)}")

    model.add_adapter(target_adapter, model.peft_config[source_adapter])
    source_marker = f".{source_adapter}."
    target_marker = f".{target_adapter}."
    with torch.no_grad():
        for name, source_param in model.named_parameters():
            if source_marker not in name:
                continue
            target_name = name.replace(source_marker, target_marker)
            target_param = model.get_parameter(target_name)
            target_param.data.copy_(source_param.data)
            target_param.requires_grad_(False)


def build_training_args(config: dict[str, Any], precision: dict[str, bool | torch.dtype], output_dir: Path, dpo_config_cls):
    train_cfg = config["training"]
    log_cfg = config["logging"]
    dpo_cfg = config["dpo"]
    model_cfg = config["model"]
    kwargs = dict(
        output_dir=str(output_dir),
        do_train=True,
        do_eval=True,
        seed=train_cfg.get("seed", 42),
        data_seed=train_cfg.get("seed", 42),
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        num_train_epochs=train_cfg["num_train_epochs"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        logging_steps=log_cfg["logging_steps"],
        save_strategy=log_cfg.get("save_strategy", "epoch"),
        eval_strategy=log_cfg.get("eval_strategy", "epoch"),
        save_total_limit=log_cfg.get("save_total_limit", 2),
        load_best_model_at_end=log_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=log_cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=log_cfg.get("greater_is_better", False),
        report_to=log_cfg.get("report_to", "none"),
        logging_first_step=True,
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        beta=dpo_cfg.get("beta", 0.1),
        loss_type=dpo_cfg.get("loss_type", "sigmoid"),
        label_smoothing=dpo_cfg.get("label_smoothing", 0.0),
        precompute_ref_log_probs=dpo_cfg.get("precompute_ref_log_probs", False),
        max_length=model_cfg["max_seq_length"],
        dataset_num_proc=config["dataset"].get("preprocessing_num_workers"),
        dataloader_num_workers=config["dataset"].get("dataloader_num_workers", 0),
        bf16=bool(precision["bf16"]),
        fp16=bool(precision["fp16"]),
        remove_unused_columns=True,
        deepspeed=get_deepspeed_config_path(config),
    )
    dpo_config_fields = getattr(dpo_config_cls, "__dataclass_fields__", {})
    if train_cfg.get("gradient_checkpointing", True) and "gradient_checkpointing_kwargs" in dpo_config_fields:
        kwargs["gradient_checkpointing_kwargs"] = train_cfg.get(
            "gradient_checkpointing_kwargs",
            {"use_reentrant": False},
        )
    if "ddp_find_unused_parameters" in dpo_config_fields:
        kwargs["ddp_find_unused_parameters"] = train_cfg.get("ddp_find_unused_parameters", False)
    if "model_adapter_name" in dpo_config_fields:
        kwargs["model_adapter_name"] = dpo_cfg.get("model_adapter_name", "default")
    if "ref_adapter_name" in dpo_config_fields:
        kwargs["ref_adapter_name"] = dpo_cfg.get("ref_adapter_name", "ref")
    return dpo_config_cls(**kwargs)


def extract_loss_points(log_history: list[dict[str, Any]], key: str) -> tuple[list[float], list[float]]:
    steps: list[float] = []
    values: list[float] = []
    for item in log_history:
        if key in item and "step" in item:
            steps.append(float(item["step"]))
            values.append(float(item[key]))
    return steps, values


def save_loss_curve(log_history: list[dict[str, Any]], output_path: Path) -> None:
    train_steps, train_loss = extract_loss_points(log_history, "loss")
    eval_steps, eval_loss = extract_loss_points(log_history, "eval_loss")
    plt.figure(figsize=(10, 6))
    if train_steps:
        plt.plot(train_steps, train_loss, label="train_loss", linewidth=2.0, color="#2563eb")
    if eval_steps:
        plt.plot(eval_steps, eval_loss, label="eval_loss", linewidth=2.0, marker="o", color="#dc2626")
    plt.title("TRL DPO Loss Curve")
    plt.xlabel("Global Step")
    plt.ylabel("Loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


DPO_METRIC_NAMES = [
    "loss",
    "eval_loss",
    "rewards/chosen",
    "rewards/rejected",
    "rewards/accuracies",
    "rewards/margins",
    "eval_rewards/chosen",
    "eval_rewards/rejected",
    "eval_rewards/accuracies",
    "eval_rewards/margins",
]


class JsonLogCallback(TrainerCallback):
    """Print Trainer logs as one flushable JSON line per logging event."""

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        if not state.is_world_process_zero or not logs:
            return
        payload = {
            "event": "trainer_log",
            "step": state.global_step,
            "epoch": state.epoch,
            **logs,
        }
        print(json.dumps(payload, ensure_ascii=False), flush=True)


def extract_dpo_metric_records(log_history: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Keep scalar DPO metrics that are useful for reports and debugging."""
    records: list[dict[str, float]] = []
    for item in log_history:
        if "step" not in item:
            continue
        record: dict[str, float] = {"step": float(item["step"])}
        if "epoch" in item:
            record["epoch"] = float(item["epoch"])
        for metric_name in DPO_METRIC_NAMES:
            if metric_name in item:
                record[metric_name] = float(item[metric_name])
        if len(record) > 1:
            records.append(record)
    return records


def save_dpo_metric_logs(log_history: list[dict[str, Any]], jsonl_path: Path, csv_path: Path) -> None:
    """Export DPO metrics from trainer_state.log_history as JSONL and CSV."""
    records = extract_dpo_metric_records(log_history)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")

    fieldnames = ["step", "epoch", *DPO_METRIC_NAMES]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fieldnames})


def save_dpo_reward_curve(log_history: list[dict[str, Any]], output_path: Path) -> None:
    """Plot chosen/rejected rewards, reward accuracy, and reward margin."""
    chosen_steps, chosen_rewards = extract_loss_points(log_history, "rewards/chosen")
    rejected_steps, rejected_rewards = extract_loss_points(log_history, "rewards/rejected")
    accuracy_steps, accuracies = extract_loss_points(log_history, "rewards/accuracies")
    margin_steps, margins = extract_loss_points(log_history, "rewards/margins")

    plt.figure(figsize=(11, 7))
    if chosen_steps:
        plt.plot(chosen_steps, chosen_rewards, label="rewards/chosen", linewidth=2.0, color="#16a34a")
    if rejected_steps:
        plt.plot(rejected_steps, rejected_rewards, label="rewards/rejected", linewidth=2.0, color="#dc2626")
    if margin_steps:
        plt.plot(margin_steps, margins, label="rewards/margins", linewidth=2.0, color="#7c3aed")
    if accuracy_steps:
        ax = plt.gca()
        ax2 = ax.twinx()
        ax2.plot(accuracy_steps, accuracies, label="rewards/accuracies", linewidth=2.0, color="#f59e0b")
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0.0, 1.0)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best")
    else:
        plt.legend()

    plt.title("TRL DPO Reward Metrics")
    plt.xlabel("Global Step")
    plt.ylabel("Reward / Margin")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def strip_thinking_tags(text: str) -> str:
    text = re.sub(r"(?s)<think>.*?</think>", "", text)
    return text.strip()


def generate_samples(trainer, tokenizer, eval_dataset, output_path: Path, max_new_tokens: int, num_examples: int) -> None:
    model = trainer.model
    model.eval()
    model_device = next(model.parameters()).device
    samples = []

    for example in eval_dataset.select(range(min(num_examples, len(eval_dataset)))):
        prompt_messages = example["prompt"]
        chosen = example["chosen"][0]["content"] if example["chosen"] else ""
        rejected = example["rejected"][0]["content"] if example["rejected"] else ""
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model_device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        output_ids = generated[0][inputs["input_ids"].shape[-1] :]
        prediction = strip_thinking_tags(tokenizer.decode(output_ids, skip_special_tokens=True))
        samples.append(
            {
                "id": example.get("id"),
                "prompt_messages": prompt_messages,
                "chosen": chosen,
                "rejected": rejected,
                "prediction": prediction,
                "reference": example.get("reference"),
                "score_delta": (example.get("metadata") or {}).get("score_delta"),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(samples, file_obj, ensure_ascii=False, indent=2)


def reset_cuda_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def get_cuda_memory_summary() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "cuda_device": None,
            "cuda_peak_allocated_mb": None,
            "cuda_peak_reserved_mb": None,
            "cuda_current_allocated_mb": None,
            "cuda_current_reserved_mb": None,
        }
    torch.cuda.synchronize()
    device = torch.cuda.current_device()
    mb = 1024 * 1024
    return {
        "cuda_available": True,
        "cuda_device": torch.cuda.get_device_name(device),
        "cuda_device_index": device,
        "cuda_peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / mb, 2),
        "cuda_peak_reserved_mb": round(torch.cuda.max_memory_reserved(device) / mb, 2),
        "cuda_current_allocated_mb": round(torch.cuda.memory_allocated(device) / mb, 2),
        "cuda_current_reserved_mb": round(torch.cuda.memory_reserved(device) / mb, 2),
    }


def get_parameter_summary(model) -> dict[str, int | float]:
    total_params = 0
    trainable_params = 0
    for parameter in model.parameters():
        num_params = parameter.numel()
        total_params += num_params
        if parameter.requires_grad:
            trainable_params += num_params
    trainable_percent = (trainable_params / total_params * 100) if total_params else 0.0
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": total_params - trainable_params,
        "trainable_percent": round(trainable_percent, 6),
    }


def save_run_summary(
    config: dict[str, Any],
    trainer,
    output_path: Path,
    memory_summary: dict[str, Any],
    parameter_summary: dict[str, int | float],
) -> None:
    summary = {
        "base_model_name_or_path": config["model"]["model_name_or_path"],
        "sft_adapter_path": config["model"]["sft_adapter_path"],
        "dpo_beta": config["dpo"].get("beta", 0.1),
        "dpo_loss_type": config["dpo"].get("loss_type", "sigmoid"),
        "model_adapter_name": config["dpo"].get("model_adapter_name", "default"),
        "ref_adapter_name": config["dpo"].get("ref_adapter_name", "ref"),
        "train_samples": len(trainer.train_dataset),
        "eval_samples": len(trainer.eval_dataset) if trainer.eval_dataset is not None else 0,
        "global_step": trainer.state.global_step,
        "best_metric": trainer.state.best_metric,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "epoch": trainer.state.epoch,
        "memory": memory_summary,
        "parameters": parameter_summary,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)


def main() -> None:
    ensure_utf8_mode()
    args = parse_args()
    config = load_config(args.config)
    validate_optional_deepspeed(config)
    distributed_ctx = get_distributed_context()
    setup_distributed_device(distributed_ctx)

    output_dir = Path(config["output"]["output_dir"]).resolve()
    artifact_dir = Path(config["output"]["artifact_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    precision = pick_precision()
    dataset = build_datasets(config)
    tokenizer = build_tokenizer(config["model"]["model_name_or_path"])
    ensure_chat_template(tokenizer, config["model"]["model_name_or_path"])
    dpo_config_cls, dpo_trainer_cls = import_trl_objects()
    dpo_config_fields = getattr(dpo_config_cls, "__dataclass_fields__", {})
    model = build_policy_model(
        config,
        precision,
        distributed_ctx,
        clone_ref_adapter="ref_adapter_name" in dpo_config_fields,
    )
    training_args = build_training_args(config, precision, output_dir, dpo_config_cls)
    trainer = dpo_trainer_cls(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )
    trainer.add_callback(JsonLogCallback())

    parameter_summary = get_parameter_summary(trainer.model)
    if trainer.is_world_process_zero():
        print(json.dumps({"parameters": parameter_summary, "beta": config["dpo"].get("beta", 0.1)}, ensure_ascii=False, indent=2))

    reset_cuda_peak_memory()
    train_result = trainer.train(resume_from_checkpoint=config["training"].get("resume_from_checkpoint"))
    memory_summary = get_cuda_memory_summary()
    train_result.metrics.update(memory_summary)
    train_result.metrics.update(parameter_summary)

    save_final_model = config.get("output", {}).get("save_final_model", not distributed_ctx["is_distributed"])
    if save_final_model:
        trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    trainer.accelerator.wait_for_everyone()

    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output_dir)
        loss_curve_path = artifact_dir / "loss_curve.png"
        dpo_metrics_jsonl_path = artifact_dir / "dpo_metrics.jsonl"
        dpo_metrics_csv_path = artifact_dir / "dpo_metrics.csv"
        reward_curve_path = artifact_dir / "dpo_reward_curve.png"
        inference_path = artifact_dir / "inference_samples.json"
        summary_path = artifact_dir / "run_summary.json"
        save_loss_curve(trainer.state.log_history, loss_curve_path)
        save_dpo_metric_logs(trainer.state.log_history, dpo_metrics_jsonl_path, dpo_metrics_csv_path)
        save_dpo_reward_curve(trainer.state.log_history, reward_curve_path)
        generate_samples(
            trainer=trainer,
            tokenizer=tokenizer,
            eval_dataset=dataset["validation"],
            output_path=inference_path,
            max_new_tokens=config["inference"]["max_new_tokens"],
            num_examples=config["inference"]["num_examples"],
        )
        save_run_summary(config, trainer, summary_path, memory_summary, parameter_summary)
        print(json.dumps(
            {
                "output_dir": str(output_dir),
                "loss_curve": str(loss_curve_path),
                "dpo_metrics_jsonl": str(dpo_metrics_jsonl_path),
                "dpo_metrics_csv": str(dpo_metrics_csv_path),
                "reward_curve": str(reward_curve_path),
                "inference_samples": str(inference_path),
                "summary": str(summary_path),
            },
            ensure_ascii=False,
            indent=2,
        ))


if __name__ == "__main__":
    main()
