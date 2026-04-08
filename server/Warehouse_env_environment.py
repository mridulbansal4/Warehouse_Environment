# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Warehouse slotting environment — OpenEnv Environment implementation.

step() returns a WarehouseObservation with reward and done set (Gym-style signals
on the observation, per OpenEnv HTTP serialization).
"""

from __future__ import annotations

import copy
import uuid
from typing import Any, Dict, List, Optional, Tuple

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from ..models import (
        ActionType,
        ConstraintViolation,
        MoveRecord,
        SKUInfo,
        WarehouseAction,
        WarehouseObservation,
        WarehouseReward,
        WarehouseState,
        ZoneInfo,
    )
    from ..tasks import (
        TaskConfig,
        check_constraints,
        compute_walk_distance,
        get_task,
    )
except ImportError:
    from models import (
        ActionType,
        ConstraintViolation,
        MoveRecord,
        SKUInfo,
        WarehouseAction,
        WarehouseObservation,
        WarehouseReward,
        WarehouseState,
        ZoneInfo,
    )
    from tasks import (
        TaskConfig,
        check_constraints,
        compute_walk_distance,
        get_task,
    )

STEP_COST = -0.01
CONSTRAINT_PENALTY = -0.15
GROUNDED_BONUS = 0.05
REDUNDANT_INSPECT_PEN = -0.03
FINAL_BONUS_FULL = 0.20
FINAL_BONUS_PARTIAL_MAX = 0.10
DISTANCE_SCALE = 0.30

DEFAULT_TASK_ID = "single_aisle_rebalance"


class WarehouseEnvironment(Environment[WarehouseAction, WarehouseObservation, WarehouseState]):
    """
    Optimise SKU placement to reduce weighted walking distance from picks to dock,
    under capacity, level (oversize), and hazardous/food aisle rules.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self.task_id: str = DEFAULT_TASK_ID
        self._episode_id: Optional[str] = None
        self._task_cfg: Optional[TaskConfig] = None
        self._zones: Dict[str, ZoneInfo] = {}
        self._skus: Dict[str, SKUInfo] = {}
        self._pending_skus: List[str] = []
        self._move_budget: int = 0
        self._steps: int = 0
        self._baseline_distance: float = 0.0
        self._move_history: List[MoveRecord] = []
        self._inspected_zones: Dict[str, int] = {}
        self._queried_skus: Dict[str, int] = {}
        self._last_non_info_action: Optional[WarehouseAction] = None
        self._done: bool = False
        self._episode_rewards: List[float] = []
        self._last_grader_score: Optional[float] = None

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="warehouse_slotting",
            description=(
                "Realistic warehouse slotting: minimise pick walk under safety and "
                "capacity constraints. Three graded tasks (easy / medium / hard)."
            ),
            version="0.1.0",
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> WarehouseObservation:
        self._reset_rubric()
        tid = (task_id or kwargs.get("task_id") or self.task_id or DEFAULT_TASK_ID)
        if isinstance(tid, str):
            self.task_id = tid.strip()
        self._episode_id = episode_id or str(uuid.uuid4())
        self._task_cfg = get_task(self.task_id)
        self._zones = copy.deepcopy(self._task_cfg.zones)
        self._skus = copy.deepcopy(self._task_cfg.skus)
        self._pending_skus = list(self._task_cfg.pending_skus)
        self._move_budget = self._task_cfg.move_budget
        self._steps = 0
        self._move_history = []
        self._inspected_zones = {}
        self._queried_skus = {}
        self._last_non_info_action = None
        self._done = False
        self._episode_rewards = []
        self._last_grader_score = None
        self._baseline_distance = compute_walk_distance(self._zones, self._skus)
        return self._build_observation(
            last_error=None,
            reward=0.0,
            done=False,
            env_context={
                "baseline_distance": self._baseline_distance,
                "walk_distance": self._baseline_distance,
            },
        )

    def step(
        self,
        action: WarehouseAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> WarehouseObservation:
        if self._done:
            return self._build_observation(
                last_error="Episode already finished. Call reset().",
                reward=0.0,
                done=True,
                env_context={"grader_score": self._last_grader_score},
            )

        self._steps += 1
        reward_model = WarehouseReward(total=0.0)
        error: Optional[str] = None

        try:
            reward_model, error = self._dispatch_action(action, reward_model)
        except Exception as exc:
            error = str(exc)

        reward_model.step_cost = STEP_COST
        reward_model.total += STEP_COST

        violations = check_constraints(self._zones, self._skus)
        if violations:
            penalty = CONSTRAINT_PENALTY * len(violations)
            reward_model.constraint_penalty = penalty
            reward_model.total += penalty

        if action.action_type == ActionType.SUBMIT_PLAN:
            self._done = True
        elif self._move_budget <= 0:
            self._done = True
        elif self._task_cfg and self._steps >= self._task_cfg.max_steps:
            self._done = True

        reward_model.total = max(-1.0, min(1.0, reward_model.total))
        self._episode_rewards.append(reward_model.total)

        env_context: Dict[str, Any] = {
            "reward_breakdown": reward_model.model_dump(),
            "violations": [
                {"rule": v[0], "sku_id": v[1], "zone_id": v[2]} for v in violations
            ],
            "walk_distance": compute_walk_distance(self._zones, self._skus),
            "baseline_distance": self._baseline_distance,
        }

        if self._done and self._task_cfg is not None:
            moves_used = self._task_cfg.move_budget - self._move_budget
            self._last_grader_score = self._task_cfg.score(
                self._zones,
                self._skus,
                self._baseline_distance,
                self._steps,
                moves_used,
            )
            env_context["grader_score"] = self._last_grader_score

        return self._build_observation(
            last_error=error,
            reward=round(reward_model.total, 4),
            done=self._done,
            env_context=env_context,
        )

    @property
    def state(self) -> WarehouseState:
        return WarehouseState(
            episode_id=self._episode_id,
            step_count=self._steps,
            task_id=self.task_id,
            episode_done=self._done,
            move_budget_remaining=self._move_budget,
            baseline_walk_distance=self._baseline_distance,
            current_walk_distance=compute_walk_distance(self._zones, self._skus),
            pending_skus=list(self._pending_skus),
            zones=dict(self._zones),
            skus=dict(self._skus),
            episode_rewards=list(self._episode_rewards),
            last_grader_score=self._last_grader_score,
        )

    def compute_final_score(self) -> float:
        if self._task_cfg is None:
            return 0.0
        moves_used = self._task_cfg.move_budget - self._move_budget
        return self._task_cfg.score(
            self._zones,
            self._skus,
            self._baseline_distance,
            self._steps,
            moves_used,
        )

    def _dispatch_action(
        self,
        action: WarehouseAction,
        reward_model: WarehouseReward,
    ) -> Tuple[WarehouseReward, Optional[str]]:
        error: Optional[str] = None
        atype = action.action_type

        if atype == ActionType.INSPECT_ZONE:
            reward_model, error = self._handle_inspect(action, reward_model)
        elif atype == ActionType.QUERY_CONSTRAINT:
            reward_model, error = self._handle_query(action, reward_model)
        elif atype == ActionType.MOVE_SKU:
            reward_model, error = self._handle_move(action, reward_model)
            if error is None:
                self._last_non_info_action = action
        elif atype == ActionType.SWAP_SKUS:
            reward_model, error = self._handle_swap(action, reward_model)
            if error is None:
                self._last_non_info_action = action
        elif atype == ActionType.FREEZE_ZONE:
            reward_model, error = self._handle_freeze(action, reward_model)
        elif atype == ActionType.SUBMIT_PLAN:
            reward_model, error = self._handle_submit(action, reward_model)
        else:
            error = f"Unsupported action_type: {atype}"

        return reward_model, error

    def _handle_inspect(
        self, action: WarehouseAction, r: WarehouseReward
    ) -> Tuple[WarehouseReward, Optional[str]]:
        if not action.zone_id:
            return r, "inspect_zone requires zone_id."
        if action.zone_id not in self._zones:
            return r, f"Zone '{action.zone_id}' does not exist."

        if action.zone_id in self._inspected_zones:
            r.efficiency_penalty += REDUNDANT_INSPECT_PEN
            r.total += REDUNDANT_INSPECT_PEN
        else:
            self._inspected_zones[action.zone_id] = self._steps

        return r, None

    def _handle_query(
        self, action: WarehouseAction, r: WarehouseReward
    ) -> Tuple[WarehouseReward, Optional[str]]:
        if not action.sku_id:
            return r, "query_constraint requires sku_id."
        if action.sku_id not in self._skus:
            return r, f"SKU '{action.sku_id}' does not exist."

        if action.sku_id not in self._queried_skus:
            self._queried_skus[action.sku_id] = self._steps

        return r, None

    def _handle_move(
        self, action: WarehouseAction, r: WarehouseReward
    ) -> Tuple[WarehouseReward, Optional[str]]:
        sku_id = action.sku_to_move
        to_zone = action.target_zone

        if not sku_id or not to_zone:
            return r, "move_sku requires sku_to_move and target_zone."
        if sku_id not in self._skus:
            return r, f"SKU '{sku_id}' does not exist."
        if to_zone not in self._zones:
            return r, f"Zone '{to_zone}' does not exist."

        dest = self._zones[to_zone]
        if dest.is_frozen:
            return r, f"Zone '{to_zone}' is frozen and cannot receive moves."
        if dest.current_sku is not None and dest.current_sku != sku_id:
            return r, f"Zone '{to_zone}' is already occupied by '{dest.current_sku}'."

        sku = self._skus[sku_id]
        if sku.weight_kg > dest.capacity_kg:
            return r, (
                f"SKU '{sku_id}' ({sku.weight_kg}kg) exceeds zone '{to_zone}' "
                f"capacity ({dest.capacity_kg}kg)."
            )
        if sku.is_oversize and dest.level != "ground":
            return r, (
                f"Oversize SKU '{sku_id}' can only go to ground-level zones; "
                f"'{to_zone}' is level '{dest.level}'."
            )

        old_distance = compute_walk_distance(self._zones, self._skus)

        old_zone_id = sku.current_zone
        if old_zone_id and old_zone_id in self._zones:
            self._zones[old_zone_id].current_sku = None
        sku.current_zone = to_zone
        dest.current_sku = sku_id

        new_distance = compute_walk_distance(self._zones, self._skus)
        delta = old_distance - new_distance

        if self._baseline_distance > 0:
            dist_reward = (delta / self._baseline_distance) * DISTANCE_SCALE
        else:
            dist_reward = 0.0
        dist_reward = max(-0.3, min(0.3, dist_reward))
        r.distance_improvement += dist_reward
        r.total += dist_reward

        if (
            to_zone in self._inspected_zones
            and self._steps - self._inspected_zones[to_zone] <= 3
        ):
            r.grounded_action_bonus += GROUNDED_BONUS
            r.total += GROUNDED_BONUS

        self._move_budget -= 1
        self._move_history.append(
            MoveRecord(
                step=self._steps,
                action_type="move",
                sku_a=sku_id,
                from_zone=old_zone_id,
                to_zone=to_zone,
                reward_delta=dist_reward,
            )
        )

        if sku_id in self._pending_skus:
            self._pending_skus.remove(sku_id)

        return r, None

    def _handle_swap(
        self, action: WarehouseAction, r: WarehouseReward
    ) -> Tuple[WarehouseReward, Optional[str]]:
        sku_a_id = action.sku_a
        sku_b_id = action.sku_b

        if not sku_a_id or not sku_b_id:
            return r, "swap_skus requires sku_a and sku_b."
        if sku_a_id not in self._skus:
            return r, f"SKU '{sku_a_id}' does not exist."
        if sku_b_id not in self._skus:
            return r, f"SKU '{sku_b_id}' does not exist."

        sku_a = self._skus[sku_a_id]
        sku_b = self._skus[sku_b_id]

        zone_a_id = sku_a.current_zone
        zone_b_id = sku_b.current_zone

        if not zone_a_id:
            return r, f"SKU '{sku_a_id}' has no assigned zone; use move_sku instead."
        if not zone_b_id:
            return r, f"SKU '{sku_b_id}' has no assigned zone; use move_sku instead."

        zone_a = self._zones[zone_a_id]
        zone_b = self._zones[zone_b_id]

        if zone_a.is_frozen:
            return r, f"Zone '{zone_a_id}' is frozen."
        if zone_b.is_frozen:
            return r, f"Zone '{zone_b_id}' is frozen."

        if sku_a.weight_kg > zone_b.capacity_kg:
            return r, f"SKU '{sku_a_id}' too heavy for zone '{zone_b_id}'."
        if sku_b.weight_kg > zone_a.capacity_kg:
            return r, f"SKU '{sku_b_id}' too heavy for zone '{zone_a_id}'."
        if sku_a.is_oversize and zone_b.level != "ground":
            return r, f"Oversize SKU '{sku_a_id}' cannot go to non-ground zone '{zone_b_id}'."
        if sku_b.is_oversize and zone_a.level != "ground":
            return r, f"Oversize SKU '{sku_b_id}' cannot go to non-ground zone '{zone_a_id}'."

        old_distance = compute_walk_distance(self._zones, self._skus)

        sku_a.current_zone = zone_b_id
        sku_b.current_zone = zone_a_id
        zone_a.current_sku = sku_b_id
        zone_b.current_sku = sku_a_id

        new_distance = compute_walk_distance(self._zones, self._skus)
        delta = old_distance - new_distance

        if self._baseline_distance > 0:
            dist_reward = (delta / self._baseline_distance) * DISTANCE_SCALE
        else:
            dist_reward = 0.0
        dist_reward = max(-0.3, min(0.3, dist_reward))
        r.distance_improvement += dist_reward
        r.total += dist_reward

        self._move_budget -= 1
        self._move_history.append(
            MoveRecord(
                step=self._steps,
                action_type="swap",
                sku_a=sku_a_id,
                sku_b=sku_b_id,
                from_zone=zone_a_id,
                to_zone=zone_b_id,
                reward_delta=dist_reward,
            )
        )

        for sid in (sku_a_id, sku_b_id):
            if sid in self._pending_skus:
                self._pending_skus.remove(sid)

        return r, None

    def _handle_freeze(
        self, action: WarehouseAction, r: WarehouseReward
    ) -> Tuple[WarehouseReward, Optional[str]]:
        if not action.zone_id:
            return r, "freeze_zone requires zone_id."
        if action.zone_id not in self._zones:
            return r, f"Zone '{action.zone_id}' does not exist."

        self._zones[action.zone_id].is_frozen = True
        return r, None

    def _handle_submit(
        self, action: WarehouseAction, r: WarehouseReward
    ) -> Tuple[WarehouseReward, Optional[str]]:
        final_distance = compute_walk_distance(self._zones, self._skus)
        if self._baseline_distance > 0:
            reduction = (self._baseline_distance - final_distance) / self._baseline_distance
        else:
            reduction = 0.0

        target = self._task_cfg.target_reduction_pct if self._task_cfg else 0.0

        if reduction >= target:
            r.final_bonus = FINAL_BONUS_FULL
        elif reduction >= target * 0.5:
            ratio = (reduction - target * 0.5) / (target * 0.5)
            r.final_bonus = FINAL_BONUS_PARTIAL_MAX * ratio
        else:
            r.final_bonus = 0.0

        r.total += r.final_bonus
        return r, None

    def _build_observation(
        self,
        last_error: Optional[str],
        reward: float,
        done: bool,
        env_context: Optional[Dict[str, Any]] = None,
    ) -> WarehouseObservation:
        violations = check_constraints(self._zones, self._skus)
        cv_list = [
            ConstraintViolation(rule=v[0], sku_id=v[1], zone_id=v[2])
            for v in violations
        ]
        ctx: Dict[str, Any] = dict(env_context or {})
        ctx.setdefault("task_id", self.task_id)
        if self._task_cfg:
            ctx.setdefault("task_description", self._task_cfg.description)

        return WarehouseObservation(
            env_context=ctx,
            zones=dict(self._zones),
            skus=dict(self._skus),
            task_name=self.task_id,
            pending_skus=list(self._pending_skus),
            move_budget=self._move_budget,
            current_walk_distance=compute_walk_distance(self._zones, self._skus),
            baseline_walk_distance=self._baseline_distance,
            constraint_violations=cv_list,
            move_history=list(self._move_history),
            step_number=self._steps,
            last_action_error=last_error,
            done=done,
            reward=reward,
            metadata={"task_id": self.task_id},
        )
