#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run offline inference for a TRL LoRA SFT checkpoint.

Inference flow
1. Read the same YAML config used in training.
2. Load the saved LoRA adapter directory.
3. Load validation data and convert it from local `conversations` format to `messages`.
4. For each sample:
   - remove the final gold assistant reply
   - render the remaining history with the tokenizer's chat template
   - tokenize and generate a new reply
5. Save `prompt_messages`, `reference`, and `prediction` for easy comparison.

About KV cache
- This script explicitly enables `use_cache=True` during generation.
- In autoregressive decoding, the model generates one token at a time.
- KV cache stores the attention keys/values from previous tokens, so the model
  does not need to recompute the whole prefix at every decoding step.
- This is useful in inference because it usually speeds up generation.
- In contrast, SFT training usually disables cache because training needs
  backpropagation over full sequences, and cache is not the right optimization there.

Data shape changes
- Raw row:
  `conversations`
- After normalization:
  `messages`
- Before generation:
  `messages[:-1]` as prompt history
- After generation:
  decoded assistant text stored in `prediction`
"""


from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for adapter path, config path, and output path."""
    parser = argparse.ArgumentParser(description="Generate sample responses from a LoRA SFT adapter.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "configs" / "trl_qwen_customer_sft.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument("--adapter-path", type=str, default=None, help="Override adapter path.")
    parser.add_argument("--output", type=str, default=None, help="Override output file.")
    parser.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="Override number of validation examples to generate. Use 0 or a negative value for all examples.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    """Load YAML config."""
    with open(path, "r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def get_system_prompt(config: dict[str, Any]) -> str:
    """Read the optional shared customer-service system prompt from YAML."""
    return (config.get("prompt", {}).get("system_prompt") or "").strip()


def prepend_system_message(messages: list[dict[str, str]], system_prompt: str) -> list[dict[str, str]]:
    """Add the shared system prompt unless the sample already contains one."""
    if not system_prompt or (messages and messages[0]["role"] == "system"):
        return messages
    return [{"role": "system", "content": system_prompt}, *messages]


def normalize_messages(example: dict[str, Any], system_prompt: str = "") -> dict[str, Any]:
    """Convert one local sample into TRL/Transformers chat-message format."""
    messages = [{"role": item["from"], "content": item["value"]} for item in example["conversations"]]
    messages = prepend_system_message(messages, system_prompt)
    return {"messages": messages}


def strip_thinking_tags(text: str) -> str:
    """Remove Qwen thinking blocks from generated replies."""
    text = re.sub(r"(?s)<think>.*?</think>", "", text)
    return text.strip()


def main() -> None:
    """Load a trained adapter and generate a few sample responses."""
    args = parse_args()
    config = load_config(args.config)
    adapter_path = Path(args.adapter_path or config["output"]["output_dir"]).resolve()
    output_path = Path(args.output or Path(config["output"]["artifact_dir"]) / "inference_samples.json").resolve()

    # The tokenizer is loaded from the saved adapter folder so it stays aligned
    # with any special-token settings used during training.
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use the most suitable dtype available on the current machine.
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        torch_dtype = torch.float32

    # `AutoPeftModelForCausalLM` restores "base model + LoRA adapter" together.
    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_path,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    model_device = next(model.parameters()).device

    # Validation data is reused here because it already represents held-out dialogue cases.
    dataset = load_dataset("json", data_files={"validation": config["dataset"]["val_file"]})
    # conversations -> messages
    dataset = dataset["validation"].map(
        normalize_messages,
        remove_columns=dataset["validation"].column_names,
        fn_kwargs={"system_prompt": get_system_prompt(config)},
    )

    samples = []
    configured_num_examples = args.num_examples
    if configured_num_examples is None:
        configured_num_examples = config["inference"].get("num_examples")
    if configured_num_examples is None or configured_num_examples <= 0:
        num_examples = len(dataset)
    else:
        num_examples = min(configured_num_examples, len(dataset))

    print(
        json.dumps(
            {
                "validation_samples": len(dataset),
                "generated_samples": num_examples,
                "output": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    for example in dataset.select(range(num_examples)):
        # The final assistant turn is the gold answer, so we hold it out for comparison.
        prompt_messages = example["messages"][:-1]
        reference = example["messages"][-1]["content"]

        # Convert structured turns into the exact chat prompt string expected by the model.
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # Tokenize prompt text into model input ids.
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model_device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=config["inference"]["max_new_tokens"],
                do_sample=False,
                # Explicitly enable KV cache during generation.
                # This is the standard inference optimization for causal LMs:
                # previous attention key/value states are reused token by token.
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        # Keep only the newly generated continuation, excluding the prompt itself.
        output_ids = generated[0][inputs["input_ids"].shape[-1] :]
        prediction = strip_thinking_tags(tokenizer.decode(output_ids, skip_special_tokens=True))
        samples.append(
            {
                # Prompt history actually fed to the model.
                "prompt_messages": prompt_messages,
                # Ground-truth final assistant reply from the dataset.
                "reference": reference,
                # Model-generated reply after LoRA SFT.
                "prediction": prediction,
            }
        )

    # Save the comparison file for qualitative evaluation or screenshots in the report.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(samples, file_obj, ensure_ascii=False, indent=2)

    print(json.dumps({"output": str(output_path), "num_samples": len(samples)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
