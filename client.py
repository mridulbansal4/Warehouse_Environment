# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Client for the warehouse slotting OpenEnv server (WebSocket session)."""

from __future__ import annotations

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import WarehouseAction, WarehouseObservation, WarehouseState
except ImportError:
    from models import WarehouseAction, WarehouseObservation, WarehouseState


class WarehouseEnv(EnvClient[WarehouseAction, WarehouseObservation, WarehouseState]):
    """
    WebSocket client for warehouse slotting. Use ``reset(task_id=...)`` to pick a task:

    - ``single_aisle_rebalance`` (easy)
    - ``cross_aisle_constrained`` (medium)
    - ``seasonal_changeover`` (hard)
    """

    def _step_payload(self, action: WarehouseAction) -> Dict[str, Any]:
        return action.model_dump(mode="json")

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[WarehouseObservation]:
        obs_raw = payload.get("observation") or {}
        merged: Dict[str, Any] = {
            **obs_raw,
            "done": payload.get("done", False),
            "reward": payload.get("reward"),
            "metadata": obs_raw.get("metadata") or {},
        }
        observation = WarehouseObservation.model_validate(merged)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> WarehouseState:
        return WarehouseState.model_validate(payload)
