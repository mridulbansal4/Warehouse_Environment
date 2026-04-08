# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Task definitions for the warehouse slotting environment.

Each task includes a deterministic grader (TaskConfig.score) in [0.0, 1.0].
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from .models import SKUInfo, ZoneInfo
except ImportError:
    from models import SKUInfo, ZoneInfo


def compute_walk_distance(
    zones: Dict[str, ZoneInfo],
    skus: Dict[str, SKUInfo],
) -> float:
    """Weighted walk distance: sum pick_frequency * distance_to_dock for assigned SKUs."""
    total = 0.0
    for sku in skus.values():
        if sku.current_zone and sku.current_zone in zones:
            zone = zones[sku.current_zone]
            d = zone.distance_to_dock
            total += sku.pick_frequency * d
    return round(total, 4)


def check_constraints(
    zones: Dict[str, ZoneInfo],
    skus: Dict[str, SKUInfo],
) -> List[Tuple[str, Optional[str], Optional[str]]]:
    violations: List[Tuple[str, Optional[str], Optional[str]]] = []

    for sku in skus.values():
        if not sku.current_zone:
            continue
        zone = zones.get(sku.current_zone)
        if zone is None:
            continue

        if sku.is_oversize and zone.level != "ground":
            violations.append(
                (
                    f"Oversize SKU {sku.sku_id} must be at ground level, "
                    f"but is at {zone.level}.",
                    sku.sku_id,
                    zone.zone_id,
                )
            )

        if sku.weight_kg > zone.capacity_kg:
            violations.append(
                (
                    f"SKU {sku.sku_id} ({sku.weight_kg}kg) exceeds zone "
                    f"{zone.zone_id} capacity ({zone.capacity_kg}kg).",
                    sku.sku_id,
                    zone.zone_id,
                )
            )

    hazardous_aisles: set = set()
    food_aisles: set = set()
    haz_skus: Dict[str, str] = {}
    food_skus: Dict[str, str] = {}

    for sku in skus.values():
        if not sku.current_zone:
            continue
        zone = zones.get(sku.current_zone)
        if zone is None:
            continue
        if sku.is_hazardous:
            hazardous_aisles.add(zone.aisle)
            haz_skus[sku.sku_id] = zone.aisle
        if sku.is_food:
            food_aisles.add(zone.aisle)
            food_skus[sku.sku_id] = zone.aisle

    shared = hazardous_aisles & food_aisles
    if shared:
        for sku_id, aisle in haz_skus.items():
            if aisle in shared:
                violations.append(
                    (
                        f"Hazardous SKU {sku_id} shares aisle {aisle} with a food SKU.",
                        sku_id,
                        None,
                    )
                )

    return violations


def _zone(
    zone_id: str,
    *,
    aisle: str,
    level: str,
    capacity_kg: float,
    distance_to_dock: float,
    sku: Optional[str] = None,
) -> ZoneInfo:
    return ZoneInfo(
        zone_id=zone_id,
        aisle=aisle,
        aisle_id=aisle,
        level=level,
        capacity_kg=capacity_kg,
        current_sku=sku,
        distance_to_dock=distance_to_dock,
        walk_meters=distance_to_dock,
        is_frozen=False,
    )


@dataclass
class TaskConfig:
    task_id: str
    display_name: str
    description: str
    max_steps: int
    move_budget: int
    target_reduction_pct: float
    zones: Dict[str, ZoneInfo] = field(default_factory=dict)
    skus: Dict[str, SKUInfo] = field(default_factory=dict)
    pending_skus: List[str] = field(default_factory=list)

    def score(
        self,
        final_zones: Dict[str, ZoneInfo],
        final_skus: Dict[str, SKUInfo],
        baseline_distance: float,
        steps_used: int,
        move_budget_used: int,
    ) -> float:
        if baseline_distance <= 0:
            return 0.0

        final_distance = compute_walk_distance(final_zones, final_skus)
        violations = check_constraints(final_zones, final_skus)

        reduction = (baseline_distance - final_distance) / baseline_distance
        distance_score = min(max(reduction / self.target_reduction_pct, 0.0), 1.0) * 0.40

        max_violations = max(1, len(final_skus))
        violation_ratio = len(violations) / max_violations
        constraint_score = max(0.0, 1.0 - violation_ratio) * 0.40

        budget = self.max_steps
        efficiency = 1.0 - (steps_used / budget) if budget > 0 else 0.0
        efficiency_score = max(0.0, efficiency) * 0.20

        total = distance_score + constraint_score + efficiency_score
        return round(min(max(total, 0.0), 1.0), 4)


def build_single_aisle_task() -> TaskConfig:
    zones: Dict[str, ZoneInfo] = {}
    for i in range(1, 9):
        zone_id = f"A{i}"
        zones[zone_id] = _zone(
            zone_id,
            aisle="A",
            level="ground",
            capacity_kg=500.0,
            distance_to_dock=float(i - 1) * 5.0,
        )

    initial_placement = {
        "SKU_001": ("A7", 80.0, 50.0, False, False),
        "SKU_002": ("A8", 60.0, 40.0, False, False),
        "SKU_003": ("A1", 30.0, 5.0, False, False),
        "SKU_004": ("A2", 25.0, 8.0, False, False),
        "SKU_005": ("A6", 70.0, 35.0, False, False),
    }

    skus: Dict[str, SKUInfo] = {}
    for sku_id, (zone, weight, freq, haz, over) in initial_placement.items():
        skus[sku_id] = SKUInfo(
            sku_id=sku_id,
            weight_kg=weight,
            pick_frequency=freq,
            is_hazardous=haz,
            is_oversize=over,
            requires_cold=False,
            current_zone=zone,
        )
        zones[zone].current_sku = sku_id

    return TaskConfig(
        task_id="single_aisle_rebalance",
        display_name="Single Aisle Rebalance",
        description=(
            "Rebalance 5 SKUs in a single 8-zone aisle so that high-velocity items "
            "sit closest to the dispatch dock, reducing total pick-walk distance."
        ),
        max_steps=20,
        move_budget=6,
        target_reduction_pct=0.30,
        zones=zones,
        skus=skus,
        pending_skus=["SKU_001", "SKU_002", "SKU_005"],
    )


def build_cross_aisle_task() -> TaskConfig:
    zones: Dict[str, ZoneInfo] = {}
    level_map = {
        1: "ground",
        2: "ground",
        3: "ground",
        4: "mid",
        5: "mid",
        6: "mid",
        7: "top",
        8: "top",
    }

    for aisle_idx, aisle in enumerate(["A", "B", "C"]):
        aisle_offset = aisle_idx * 20.0
        for i in range(1, 9):
            zone_id = f"{aisle}{i}"
            lv = level_map[i]
            zones[zone_id] = _zone(
                zone_id,
                aisle=aisle,
                level=lv,
                capacity_kg=300.0 if lv == "ground" else 150.0,
                distance_to_dock=aisle_offset + float(i - 1) * 4.0,
            )

    sku_specs = [
        ("SKU_A01", "A7", 80.0, 55.0, False, False, False),
        ("SKU_A02", "B7", 120.0, 48.0, False, True, False),
        ("SKU_A03", "C5", 40.0, 12.0, True, False, False),
        ("SKU_A04", "A3", 60.0, 30.0, False, False, True),
        ("SKU_A05", "B4", 90.0, 60.0, False, False, False),
        ("SKU_A06", "C8", 50.0, 20.0, False, False, False),
        ("SKU_A07", "A8", 70.0, 44.0, False, False, False),
        ("SKU_A08", "B6", 35.0, 8.0, False, False, False),
    ]

    skus: Dict[str, SKUInfo] = {}
    for sku_id, zone, weight, freq, haz, over, food in sku_specs:
        skus[sku_id] = SKUInfo(
            sku_id=sku_id,
            weight_kg=weight,
            pick_frequency=freq,
            is_hazardous=haz,
            is_oversize=over,
            is_food=food,
            requires_cold=False,
            current_zone=zone,
        )
        zones[zone].current_sku = sku_id

    return TaskConfig(
        task_id="cross_aisle_constrained",
        display_name="Cross-Aisle Constrained Slotting",
        description=(
            "Reassign 8 SKUs across a 3-aisle, 24-zone warehouse. "
            "Honour oversize (ground-only), weight, and hazardous–food separation "
            "while minimising weighted walk distance."
        ),
        max_steps=35,
        move_budget=14,
        target_reduction_pct=0.25,
        zones=zones,
        skus=skus,
        pending_skus=[
            "SKU_A01",
            "SKU_A02",
            "SKU_A03",
            "SKU_A04",
            "SKU_A05",
            "SKU_A06",
            "SKU_A07",
            "SKU_A08",
        ],
    )


def build_seasonal_changeover_task() -> TaskConfig:
    zones: Dict[str, ZoneInfo] = {}
    level_map = {
        1: "ground",
        2: "ground",
        3: "ground",
        4: "mid",
        5: "mid",
        6: "mid",
        7: "mid",
        8: "mid",
        9: "top",
        10: "top",
        11: "top",
        12: "top",
    }

    for aisle_idx, aisle in enumerate(["A", "B", "C", "D"]):
        aisle_offset = aisle_idx * 25.0
        for i in range(1, 13):
            zone_id = f"{aisle}{i:02d}"
            lv = level_map[i]
            cap = 400.0 if lv == "ground" else 200.0
            zones[zone_id] = _zone(
                zone_id,
                aisle=aisle,
                level=lv,
                capacity_kg=cap,
                distance_to_dock=aisle_offset + float(i - 1) * 3.5,
            )

    existing_specs = [
        ("REG_01", "A09", 80, 12, False, False),
        ("REG_02", "A10", 60, 45, False, False),
        ("REG_03", "B09", 40, 38, False, False),
        ("REG_04", "B10", 55, 8, False, False),
        ("REG_05", "C09", 90, 50, False, False),
        ("REG_06", "C10", 70, 22, False, False),
        ("REG_07", "D09", 65, 30, False, False),
        ("REG_08", "D10", 45, 5, False, False),
        ("REG_09", "A11", 30, 55, False, False),
        ("REG_10", "B11", 50, 48, False, False),
        ("REG_11", "C11", 80, 15, False, False),
        ("REG_12", "D11", 60, 60, False, False),
        ("REG_13", "A12", 35, 3, False, False),
        ("REG_14", "B12", 75, 25, False, False),
        ("REG_15", "C12", 55, 10, False, False),
        ("REG_16", "D12", 40, 35, False, False),
        ("REG_17", "A05", 90, 18, False, False),
        ("REG_18", "B05", 30, 42, False, False),
        ("REG_19", "C05", 60, 7, False, False),
        ("REG_20", "D05", 80, 28, False, False),
        ("REG_21", "A06", 50, 33, False, False),
        ("REG_22", "B06", 40, 16, False, False),
        ("REG_23", "C06", 70, 52, False, False),
        ("REG_24", "D06", 35, 9, False, False),
        ("REG_25", "A07", 45, 40, False, False),
    ]

    seasonal_specs = [
        ("SEA_01", None, 75, 90, False, False),
        ("SEA_02", None, 50, 85, False, False),
        ("SEA_03", None, 60, 70, False, False),
        ("SEA_04", None, 80, 65, False, True),
        ("SEA_05", None, 40, 60, False, False),
        ("SEA_06", None, 55, 55, False, False),
        ("SEA_07", None, 90, 50, False, True),
        ("SEA_08", None, 35, 45, True, False),
        ("SEA_09", None, 65, 40, False, False),
        ("SEA_10", None, 45, 35, False, False),
        ("SEA_11", None, 70, 30, False, False),
        ("SEA_12", None, 50, 25, False, False),
        ("SEA_13", None, 80, 20, False, False),
        ("SEA_14", None, 30, 15, True, False),
        ("SEA_15", None, 60, 10, False, False),
    ]

    skus: Dict[str, SKUInfo] = {}
    for sku_id, zone, weight, freq, haz, over in existing_specs + seasonal_specs:
        skus[sku_id] = SKUInfo(
            sku_id=sku_id,
            weight_kg=float(weight),
            pick_frequency=float(freq),
            is_hazardous=haz,
            is_oversize=over,
            requires_cold=False,
            current_zone=zone,
        )
        if zone:
            zones[zone].current_sku = sku_id

    pending = [s[0] for s in seasonal_specs] + [
        s[0] for s in existing_specs if s[4] or s[5]
    ]

    return TaskConfig(
        task_id="seasonal_changeover",
        display_name="Seasonal Changeover",
        description=(
            "Reorganise a 48-zone warehouse for peak season: slot 15 incoming seasonal "
            "SKUs and reposition existing stock within a 50-move budget while respecting "
            "oversize and hazardous constraints."
        ),
        max_steps=60,
        move_budget=50,
        target_reduction_pct=0.20,
        zones=zones,
        skus=skus,
        pending_skus=pending,
    )


TASK_BUILDERS = {
    "single_aisle_rebalance": build_single_aisle_task,
    "cross_aisle_constrained": build_cross_aisle_task,
    "seasonal_changeover": build_seasonal_changeover_task,
}

TASK_DIFFICULTY = {
    "single_aisle_rebalance": "easy",
    "cross_aisle_constrained": "medium",
    "seasonal_changeover": "hard",
}


def get_task(task_id: str) -> TaskConfig:
    tid = task_id.strip().lower()
    if tid not in TASK_BUILDERS:
        raise ValueError(
            f"Unknown task '{task_id}'. Available: {list(TASK_BUILDERS.keys())}"
        )
    return copy.deepcopy(TASK_BUILDERS[tid]())
