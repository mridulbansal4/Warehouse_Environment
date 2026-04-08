# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pydantic models for warehouse slotting (OpenEnv Action / Observation / State)."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, ConfigDict, Field


class ActionType(str, Enum):
    INSPECT_ZONE = "inspect_zone"
    QUERY_CONSTRAINT = "query_constraint"
    MOVE_SKU = "move_sku"
    SWAP_SKUS = "swap_skus"
    FREEZE_ZONE = "freeze_zone"
    SUBMIT_PLAN = "submit_plan"


class ZoneInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    zone_id: str
    aisle: str = "A"
    level: str = "ground"
    capacity_kg: float = 100.0
    current_sku: Optional[str] = None
    is_frozen: bool = False
    distance_to_dock: float = 0.0
    walk_meters: float = 0.0
    aisle_id: str = "A"


class SKUInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    sku_id: str
    weight_kg: float = 1.0
    pick_frequency: float = 1.0
    is_hazardous: bool = False
    is_oversize: bool = False
    is_food: bool = False
    requires_cold: bool = False
    current_zone: Optional[str] = None
    category: str = "general"


class ConstraintViolation(BaseModel):
    rule: str
    sku_id: Optional[str] = None
    zone_id: Optional[str] = None


class MoveRecord(BaseModel):
    step: int
    action_type: str
    sku_a: Optional[str] = None
    sku_b: Optional[str] = None
    from_zone: Optional[str] = None
    to_zone: Optional[str] = None
    reward_delta: float = 0.0


class WarehouseReward(BaseModel):
    model_config = ConfigDict(extra="allow")

    total: float = 0.0
    step_cost: float = 0.0
    constraint_penalty: float = 0.0
    distance_improvement: float = 0.0
    grounded_action_bonus: float = 0.0
    efficiency_penalty: float = 0.0
    final_bonus: float = 0.0


class WarehouseAction(Action):
    """One warehouse operations action (inspect, move, swap, submit, …)."""

    action_type: ActionType
    zone_id: Optional[str] = None
    sku_id: Optional[str] = None
    sku_to_move: Optional[str] = None
    target_zone: Optional[str] = None
    sku_a: Optional[str] = None
    sku_b: Optional[str] = None


class WarehouseObservation(Observation):
    """Full observation: layout, constraints, progress, plus OpenEnv reward/done/metadata."""

    # Included in HTTP/WS payloads (unlike ``metadata``, which OpenEnv strips for clients).
    env_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Reward breakdown, violations summary, distances, grader_score when done",
    )
    zones: Dict[str, ZoneInfo] = Field(default_factory=dict)
    skus: Dict[str, SKUInfo] = Field(default_factory=dict)
    task_name: str = ""
    pending_skus: List[str] = Field(default_factory=list)
    move_budget: int = 0
    current_walk_distance: float = 0.0
    baseline_walk_distance: float = 0.0
    constraint_violations: List[ConstraintViolation] = Field(default_factory=list)
    move_history: List[MoveRecord] = Field(default_factory=list)
    step_number: int = 0
    last_action_error: Optional[str] = None


class WarehouseState(State):
    """Serializable environment state (HTTP / WebSocket / grading)."""

    task_id: str = ""
    episode_done: bool = False
    move_budget_remaining: int = 0
    baseline_walk_distance: float = 0.0
    current_walk_distance: float = 0.0
    pending_skus: List[str] = Field(default_factory=list)
    zones: Dict[str, ZoneInfo] = Field(default_factory=dict)
    skus: Dict[str, SKUInfo] = Field(default_factory=dict)
    episode_rewards: List[float] = Field(default_factory=list)
    last_grader_score: Optional[float] = None
