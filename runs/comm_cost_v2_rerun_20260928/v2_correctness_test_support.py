"""Small synthetic inputs for the Stage C trainer correctness regressions."""

from __future__ import annotations

import numpy as np
import torch

from onpolicy import joint_log_prob


class Collector:
    def __init__(self, records):
        self.records = records


def record(controller, step: int, *, value: float | None = None,
           cost_value: float | None = None, action: int = 1,
           features: np.ndarray | None = None) -> dict:
    features = (np.zeros((1, 19), dtype=np.float32) if features is None
                else np.asarray(features, dtype=np.float32).reshape(1, 19))
    matrix = torch.as_tensor(features)
    with torch.no_grad():
        logits, values = controller.forward(matrix)
        costs = controller.cost_values(matrix)
        log_prob = joint_log_prob(logits, torch.tensor([action]), torch.tensor([1.0]))
    return {
        "step": step,
        "targets": ["task"],
        "actions": {"task": action},
        "actionable": ["task"],
        "features": features.tolist(),
        "log_prob": float(log_prob),
        "value": float(values.mean()) if value is None else value,
        "cost_value": float(costs.mean()) if cost_value is None else cost_value,
    }


def episode(records, *, steps: int, cost_stream=None, task_stream=None,
            lam: float = 1.0) -> dict:
    return {
        "collector": Collector(records),
        "steps": steps,
        "cost_stream": cost_stream or {},
        "task_stream": task_stream or {},
        "lambda": lam,
        "episode_index": 0,
    }
