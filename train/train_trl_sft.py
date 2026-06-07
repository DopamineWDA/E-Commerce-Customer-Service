#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Use TRL's SFTTrainer to finetune JD customer-service dialogue data with LoRA.

Training flow
1. Read the YAML config: model path, dataset path, LoRA settings, training hyperparameters.
2. Load `train.jsonl` / `val.jsonl`.
3. Convert each sample from the local ShareGPT-like format:
   {
     "conversations": [{"from": "user", "value": "..."}, {"from": "assistant", "value": "..."}]
   }
   into TRL's conversational format:
   {
     "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
   }
4. Load tokenizer + base model.
5. Build LoRA config and SFT config, then create `trl.SFTTrainer`.
6. Let TRL apply the chat template and tokenize the `messages` field internally.
7. Train and evaluate.
8. Export model/adapters, trainer state, loss curve, and a few inference samples.

Optional distributed training
- This script can optionally pass a DeepSpeed config into `trl.SFTTrainer`.
- If a ZeRO config is enabled, the distributed launcher / Trainer is responsible
  for device placement, so we do not use `device_map="auto"` in that case.

Data shape changes
- Raw file row:
  `id/session_id/conversations`
- After `normalize_messages()`:
  `messages`
- Inside TRL:
  `messages` -> chat-formatted text -> token ids -> loss on assistant tokens
"""

from __future__ import annotations

import argparse
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
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


def import_trl_objects():
    """Import TRL lazily after the process is guaranteed to run in UTF-8 mode."""
    ensure_utf8_mode()
    from trl import SFTConfig, SFTTrainer

    return SFTConfig, SFTTrainer


def ensure_utf8_mode() -> None:
    # TRL reads bundled chat-template files with the process default text encoding.
    # On Windows, re-exec in UTF-8 mode to avoid GBK decode errors during import.
    if not sys.flags.utf8_mode:
        os.environ["PYTHONUTF8"] = "1"
        os.execv(sys.executable, [sys.executable, *sys.argv])


def parse_args() -> argparse.Namespace:
    """Read only one argument: the path to the YAML config file."""
    parser = argparse.ArgumentParser(description="Train TRL SFT with LoRA on JD chat data.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "configs" / "trl_qwen_customer_sft.yaml"),
        help="Path to YAML config.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    """Load a YAML config into a Python dict.

    We also store the config directory so relative paths inside the YAML can be
    resolved later, for example a DeepSpeed JSON file.
    """
    with open(path, "r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj)
    config["__config_dir__"] = str(Path(path).resolve().parent)
    return config


def resolve_from_config_dir(config: dict[str, Any], maybe_path: str | None) -> str | None:
    """Resolve an optional path relative to the YAML file location."""
    if not maybe_path:
        return None
    candidate = Path(maybe_path)
    if candidate.is_absolute():
        return str(candidate)
    return str((Path(config["__config_dir__"]) / candidate).resolve())


def get_deepspeed_config_path(config: dict[str, Any]) -> str | None:
    """Return the resolved DeepSpeed config path if enabled in YAML."""
    distributed_cfg = config.get("distributed", {})
    return resolve_from_config_dir(config, distributed_cfg.get("deepspeed"))


def get_system_prompt(config: dict[str, Any]) -> str:
    """Read the optional shared customer-service system prompt from YAML."""
    return (config.get("prompt", {}).get("system_prompt") or "").strip()


def prepend_system_message(messages: list[dict[str, str]], system_prompt: str) -> list[dict[str, str]]:
    """Add the shared system prompt unless the sample already contains one."""
    if not system_prompt or (messages and messages[0]["role"] == "system"):
        return messages
    return [{"role": "system", "content": system_prompt}, *messages]


def is_deepspeed_enabled(config: dict[str, Any]) -> bool:
    """Whether this run enables DeepSpeed through the YAML config."""
    return get_deepspeed_config_path(config) is not None


def validate_optional_deepspeed(config: dict[str, Any]) -> None:
    """Fail early with a clear message if YAML enables DeepSpeed but the package is missing."""
    if not is_deepspeed_enabled(config):
        return

    try:
        importlib.metadata.version("deepspeed")
    except importlib.metadata.PackageNotFoundError as exc:
        ds_path = get_deepspeed_config_path(config)
        raise RuntimeError(
            "DeepSpeed is enabled in the YAML config, but the `deepspeed` package is not installed. "
            f"Configured file: {ds_path}. "
            "Install DeepSpeed first, or set `distributed.deepspeed:` to empty to disable it."
        ) from exc


def get_distributed_context() -> dict[str, int | bool]:
    """Read launcher-provided distributed environment variables."""
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size > 1
    return {
        "local_rank": local_rank,
        "rank": rank,
        "world_size": world_size,
        "is_distributed": is_distributed,
        "is_main_process": rank == 0,
    }


def setup_distributed_device(distributed_ctx: dict[str, int | bool]) -> None:
    """Bind the current process to its launcher-assigned GPU when using torchrun/deepspeed."""
    if not distributed_ctx["is_distributed"]:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed launch was detected, but CUDA is not available.")
    local_rank = int(distributed_ctx["local_rank"])
    if local_rank < 0:
        raise RuntimeError("Distributed launch was detected, but LOCAL_RANK is missing.")
    torch.cuda.set_device(local_rank)


def normalize_messages(example: dict[str, Any], system_prompt: str = "") -> dict[str, Any]:
    """Convert one local training sample into TRL's expected conversational format.

    Before:
    {
      "conversations": [{"from": "user", "value": "..."}, {"from": "assistant", "value": "..."}]
    }

    After:
    {
      "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    }
    """
    messages = [{"role": item["from"], "content": item["value"]} for item in example["conversations"]]
    messages = prepend_system_message(messages, system_prompt)
    return {"messages": messages}


def build_datasets(config: dict[str, Any]) -> DatasetDict:
    """Load train/validation jsonl files and normalize the message schema.

    Important:
    - We drop the original columns after conversion because TRL only needs `messages`.
    - We keep optional sample truncation hooks (`max_train_samples`, `max_eval_samples`)
      so experiments can be run quickly on a subset.
    """
    data_files = {
        "train": config["dataset"]["train_file"],
        "validation": config["dataset"]["val_file"],
    }
    dataset = load_dataset("json", data_files=data_files)
    remove_columns = dataset["train"].column_names
    num_proc = config["dataset"].get("preprocessing_num_workers")
    system_prompt = get_system_prompt(config)
    # Example-level transformation:
    # id/session_id/conversations -> messages
    dataset = dataset.map(
        normalize_messages,
        remove_columns=remove_columns,
        num_proc=num_proc,
        fn_kwargs={"system_prompt": system_prompt},
    )

    max_train_samples = config["dataset"].get("max_train_samples")
    if max_train_samples:
        dataset["train"] = dataset["train"].select(range(min(max_train_samples, len(dataset["train"]))))

    max_eval_samples = config["dataset"].get("max_eval_samples")
    if max_eval_samples:
        dataset["validation"] = dataset["validation"].select(range(min(max_eval_samples, len(dataset["validation"]))))

    return dataset


def pick_precision() -> dict[str, bool | torch.dtype]:
    """Choose a safe mixed-precision mode from the current hardware.

    Return values are split into:
    - `dtype`: used when loading the model
    - `bf16` / `fp16`: passed into TRL/Transformers training arguments
    """
    if not torch.cuda.is_available():
        return {"bf16": False, "fp16": False, "dtype": torch.float32}
    if torch.cuda.is_bf16_supported():
        return {"bf16": True, "fp16": False, "dtype": torch.bfloat16}
    return {"bf16": False, "fp16": True, "dtype": torch.float16}


def build_tokenizer(model_name_or_path: str):
    """Load tokenizer and make sure padding works for causal LM training."""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    # Many chat models do not define a separate pad token.
    # In causal LM training, using eos_token as pad_token is a common fallback.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Right padding is the standard choice for most causal language model trainers.
    tokenizer.padding_side = "right"
    return tokenizer


def _resolve_training_chat_template_path(model_name_or_path: str) -> Path | None:
    """Pick a vendored training chat template for common model families used in this project."""
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


def _load_training_chat_template(model_name_or_path: str) -> str | None:
    """Load the vendored training chat template text if this model family is supported."""
    template_path = _resolve_training_chat_template_path(model_name_or_path)
    if template_path is None or not template_path.is_file():
        return None
    return template_path.read_text(encoding="utf-8")


def ensure_training_chat_template(
    tokenizer,
    model_name_or_path: str,
    assistant_only_loss: bool,
) -> None:
    """Patch tokenizer.chat_template when assistant-only masking is enabled on older site-packages TRL."""
    if not assistant_only_loss:
        return

    chat_template = getattr(tokenizer, "chat_template", None)
    if not chat_template:
        raise RuntimeError(
            "assistant_only_loss=True requires the tokenizer to define a chat template, "
            f"but `{model_name_or_path}` did not provide one."
        )

    if "{% generation %}" in chat_template:
        return

    patched_template = _load_training_chat_template(model_name_or_path)
    if not patched_template:
        raise RuntimeError(
            "assistant_only_loss=True requires a training-compatible chat template with `{% generation %}` markers, "
            f"but no compatible vendored template was found for `{model_name_or_path}`."
        )

    tokenizer.chat_template = patched_template
    warnings.warn(
        "The tokenizer chat template was automatically patched for assistant-only loss compatibility. "
        "This keeps training working when Linux uses site-packages/trl instead of the vendored TRL copy.",
        stacklevel=2,
    )


def build_model(model_name_or_path: str, dtype: torch.dtype, use_device_map: bool = True):
    """Load the base causal LM that LoRA adapters will be attached to."""
    return AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        # `device_map="auto"` lets Transformers place the model on available GPU(s).
        device_map="auto" if use_device_map and torch.cuda.is_available() else None,
    )


def build_model_for_training(
    config: dict[str, Any],
    precision: dict[str, bool | torch.dtype],
    distributed_ctx: dict[str, int | bool],
):
    """Load the model with placement rules that match the chosen training backend.

    Single-process training:
    - use `device_map="auto"` for convenience

    Distributed / DeepSpeed training:
    - do not use `device_map="auto"`
    - bind each process to its local GPU and let the distributed stack own placement
    """
    model_name_or_path = config["model"]["model_name_or_path"]
    dtype = precision["dtype"]

    if distributed_ctx["is_distributed"] or is_deepspeed_enabled(config):
        return AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=dtype,
        )

    return build_model(model_name_or_path, dtype, use_device_map=True)


def build_training_args(
    config: dict[str, Any],
    precision: dict[str, bool | torch.dtype],
    output_dir: Path,
    sft_config_cls,
):
    """Translate our YAML config into a TRL `SFTConfig`.

    This function is the bridge between our project-style config file and the
    actual trainer arguments expected by the installed TRL version.
    """
    train_cfg = config["training"]
    log_cfg = config["logging"]
    model_cfg = config["model"]
    deepspeed_config_path = get_deepspeed_config_path(config)
    return sft_config_cls(
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
        # `max_length` is the final token length after chat template rendering.
        max_length=model_cfg["max_seq_length"],
        packing=model_cfg.get("packing", False),
        # Only compute loss on assistant replies in the dialogue.
        assistant_only_loss=train_cfg.get("assistant_only_loss", True),
        dataset_num_proc=config["dataset"].get("preprocessing_num_workers"),
        dataloader_num_workers=config["dataset"].get("dataloader_num_workers", 0),
        bf16=bool(precision["bf16"]),
        fp16=bool(precision["fp16"]),
        # We keep dataset columns ourselves and do not want Trainer to drop them unexpectedly.
        remove_unused_columns=False,
        eos_token=model_cfg.get("eos_token"),
        # If this path is set, Hugging Face Trainer initializes DeepSpeed from the JSON config.
        deepspeed=deepspeed_config_path,
    )


def build_lora_config(config: dict[str, Any]) -> LoraConfig:
    """Build the PEFT LoRA config used by TRL's SFTTrainer."""
    lora_cfg = config["lora"]
    return LoraConfig(
        task_type="CAUSAL_LM",
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg.get("bias", "none"),
        target_modules=lora_cfg["target_modules"],
    )


def extract_loss_points(log_history: list[dict[str, Any]], key: str) -> tuple[list[float], list[float]]:
    """Pull `(global_step, metric_value)` points from Hugging Face trainer logs."""
    steps: list[float] = []
    values: list[float] = []
    for item in log_history:
        if key in item and "step" in item:
            steps.append(float(item["step"]))
            values.append(float(item[key]))
    return steps, values


def save_loss_curve(log_history: list[dict[str, Any]], output_path: Path) -> None:
    """Render a PNG loss curve from `trainer.state.log_history`.

    The trainer logs multiple records over time. We only keep:
    - `loss` for training
    - `eval_loss` for validation
    """
    train_steps, train_loss = extract_loss_points(log_history, "loss")
    eval_steps, eval_loss = extract_loss_points(log_history, "eval_loss")

    plt.figure(figsize=(10, 6))
    if train_steps:
        plt.plot(train_steps, train_loss, label="train_loss", linewidth=2.0, color="#1f77b4")
    if eval_steps:
        plt.plot(eval_steps, eval_loss, label="eval_loss", linewidth=2.0, marker="o", color="#d62728")
    plt.title("TRL SFT Loss Curve")
    plt.xlabel("Global Step")
    plt.ylabel("Loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def build_prompt_from_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    # Each validation sample already contains the gold assistant answer at the end.
    # For generation, we remove that final answer and keep only the dialogue history.
    return messages[:-1]


def strip_thinking_tags(text: str) -> str:
    """Remove Qwen thinking blocks from generated inference samples."""
    text = re.sub(r"(?s)<think>.*?</think>", "", text)
    return text.strip()


def generate_samples(
    trainer,
    tokenizer,
    eval_dataset,
    output_path: Path,
    max_new_tokens: int,
    num_examples: int,
) -> None:
    """Generate a few validation-set predictions after training.

    Data flow here:
    `messages` -> prompt messages only -> tokenizer chat template -> token ids
    -> model.generate() -> decoded assistant reply

    The output file is useful for qualitative inspection in reports:
    compare `reference` vs `prediction`.
    """
    model = trainer.model
    model.eval()
    model_device = next(model.parameters()).device
    samples = []

    for example in eval_dataset.select(range(min(num_examples, len(eval_dataset)))):
        prompt_messages = build_prompt_from_messages(example["messages"])
        reference = example["messages"][-1]["content"]
        # Convert structured multi-turn messages into the exact string prompt the model expects.
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        # Tokenization happens after the chat template is rendered to plain text.
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
        # Only decode newly generated tokens, not the original prompt tokens.
        output_ids = generated[0][inputs["input_ids"].shape[-1] :]
        prediction = strip_thinking_tags(tokenizer.decode(output_ids, skip_special_tokens=True))
        samples.append(
            {
                "prompt_messages": prompt_messages,
                "reference": reference,
                "prediction": prediction,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(samples, file_obj, ensure_ascii=False, indent=2)


def reset_cuda_peak_memory() -> None:
    """Start a fresh CUDA peak-memory window for the training phase."""
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def get_cuda_memory_summary() -> dict[str, Any]:
    """Return per-process CUDA memory stats in MiB for experiment comparison."""
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
    """Count total and trainable parameters after PEFT adapters are attached."""
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
    trainer: SFTTrainer,
    output_path: Path,
    memory_summary: dict[str, Any] | None = None,
    parameter_summary: dict[str, int | float] | None = None,
) -> None:
    """Write a short machine-readable summary of the finished run."""
    summary = {
        "model_name_or_path": config["model"]["model_name_or_path"],
        "train_samples": len(trainer.train_dataset),
        "eval_samples": len(trainer.eval_dataset) if trainer.eval_dataset is not None else 0,
        "global_step": trainer.state.global_step,
        "best_metric": trainer.state.best_metric,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "epoch": trainer.state.epoch,
        "system_prompt": get_system_prompt(config),
        "memory": memory_summary or get_cuda_memory_summary(),
        "parameters": parameter_summary or get_parameter_summary(trainer.model),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)


def main() -> None:
    """Main entry point for the full SFT pipeline."""
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

    # 1. Resolve precision and load data.
    precision = pick_precision()
    dataset = build_datasets(config)

    # 2. Load tokenizer/model from the base checkpoint.
    tokenizer = build_tokenizer(config["model"]["model_name_or_path"])
    ensure_training_chat_template(
        tokenizer=tokenizer,
        model_name_or_path=config["model"]["model_name_or_path"],
        assistant_only_loss=config["training"].get("assistant_only_loss", True),
    )
    model = build_model_for_training(config, precision, distributed_ctx)

    # 3. Import TRL lazily, then translate our YAML into TRL configs.
    sft_config_cls, sft_trainer_cls = import_trl_objects()
    training_args = build_training_args(config, precision, output_dir, sft_config_cls)
    peft_config = build_lora_config(config)

    # 4. Create the TRL trainer.
    # TRL will take `messages`, apply the model chat template, tokenize, and compute SFT loss.
    trainer = sft_trainer_cls(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    parameter_summary = get_parameter_summary(trainer.model)
    if trainer.is_world_process_zero():
        print(json.dumps({"parameters": parameter_summary}, ensure_ascii=False, indent=2))

    # 5. Start training. If a checkpoint path exists in config, resume from it.
    reset_cuda_peak_memory()
    train_result = trainer.train(resume_from_checkpoint=config["training"].get("resume_from_checkpoint"))
    memory_summary = get_cuda_memory_summary()
    train_result.metrics.update(memory_summary)
    train_result.metrics.update(parameter_summary)

    # 6. Save training outputs.
    # For LoRA training, `save_model()` mainly stores the adapter weights plus config files.
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    trainer.accelerator.wait_for_everyone()

    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output_dir)

    # 7. Convert trainer logs into human-friendly artifacts.
    loss_curve_path = artifact_dir / "loss_curve.png"
    inference_path = artifact_dir / "inference_samples.json"
    summary_path = artifact_dir / "run_summary.json"
    if trainer.is_world_process_zero():
        save_loss_curve(trainer.state.log_history, loss_curve_path)
        generate_samples(
            trainer=trainer,
            tokenizer=tokenizer,
            eval_dataset=dataset["validation"],
            output_path=inference_path,
            max_new_tokens=config["inference"]["max_new_tokens"],
            num_examples=config["inference"]["num_examples"],
        )
        save_run_summary(
            config,
            trainer,
            summary_path,
            memory_summary=memory_summary,
            parameter_summary=parameter_summary,
        )

    if trainer.is_world_process_zero():
        print(json.dumps(
            {
                "output_dir": str(output_dir),
                "loss_curve": str(loss_curve_path),
                "inference_samples": str(inference_path),
                "summary": str(summary_path),
            },
            ensure_ascii=False,
            indent=2,
        ))


if __name__ == "__main__":
    main()
