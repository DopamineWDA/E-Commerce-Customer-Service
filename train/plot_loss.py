#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot loss curves from Hugging Face trainer_state.json.

This script is intentionally small and report-oriented:
1. Read `trainer_state.json` saved by Hugging Face Trainer / TRL.
2. Extract `loss` and `eval_loss` from `log_history`.
3. Plot them against `global_step`.
4. Save one PNG figure that can be used directly in the experiment report.

Data shape changes
- Input JSON:
  `trainer_state["log_history"]` is a list of logging records
- Extracted points:
  `(step, loss)` and `(step, eval_loss)`
- Output:
  a static PNG line chart
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    """Parse the trainer-state path and the output png path."""
    parser = argparse.ArgumentParser(description="Plot TRL training loss from trainer_state.json.")
    parser.add_argument("--trainer-state", type=str, required=True, help="Path to trainer_state.json.")
    parser.add_argument("--output", type=str, required=True, help="Path to output PNG.")
    return parser.parse_args()


def extract_points(log_history: list[dict[str, Any]], metric_name: str) -> tuple[list[float], list[float]]:
    """Extract one metric series from the trainer log history.

    Example:
    if `metric_name="loss"`, collect every record that looks like:
    {"loss": 1.23, "step": 100, ...}
    """
    x_values: list[float] = []
    y_values: list[float] = []
    for record in log_history:
        if metric_name in record and "step" in record:
            x_values.append(float(record["step"]))
            y_values.append(float(record[metric_name]))
    return x_values, y_values


def main() -> None:
    """Read trainer logs and render the final loss figure."""
    args = parse_args()
    trainer_state_path = Path(args.trainer_state)
    output_path = Path(args.output)

    with trainer_state_path.open("r", encoding="utf-8") as file_obj:
        trainer_state = json.load(file_obj)

    # `log_history` stores all periodic logging events emitted during training/eval.
    log_history = trainer_state.get("log_history", [])
    train_steps, train_loss = extract_points(log_history, "loss")
    eval_steps, eval_loss = extract_points(log_history, "eval_loss")

    plt.figure(figsize=(10, 6))
    if train_steps:
        plt.plot(train_steps, train_loss, label="train_loss", linewidth=2.0, color="#1f77b4")
    if eval_steps:
        plt.plot(eval_steps, eval_loss, label="eval_loss", linewidth=2.0, marker="o", color="#d62728")
    plt.title("TRL Loss Curve")
    plt.xlabel("Global Step")
    plt.ylabel("Loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
