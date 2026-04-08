# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic episode grader in [0.0, 1.0] (mirrors TaskConfig.score)."""

from __future__ import annotations

from typing import Any, Dict

try:
    from .models import SKUInfo, ZoneInfo
    from .tasks import get_task
except ImportError:
    from models import SKUInfo, ZoneInfo
    from tasks import get_task


def compute_episode_score(
    task_id: str,
    *,
    zones: Dict[str, Any],
    skus: Dict[str, Any],
    baseline_distance: float,
    steps_used: int,
    move_budget_used: int,
) -> float:
    """
    Grade a completed or partial episode from plain dicts (e.g. state snapshot).

    ``move_budget_used`` is the number of physical moves/swaps consumed
    (initial_budget - remaining), as tracked by the environment.
    """
    cfg = get_task(task_id)
    z = {k: ZoneInfo.model_validate(v) for k, v in zones.items()}
    s = {k: SKUInfo.model_validate(v) for k, v in skus.items()}
    return cfg.score(z, s, baseline_distance, steps_used, move_budget_used)
