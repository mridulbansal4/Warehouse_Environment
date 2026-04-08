#!/usr/bin/env python3
"""
inference.py — Warehouse Slotting OpenEnv Baseline
==================================================

Runs warehouse tasks (easy / medium / hard) against the OpenEnv HTTP/WebSocket server
and emits the structured log format expected by the hackathon evaluator.

This file does **not** import ``env`` (local grid). Phase 2 validation runs in a minimal
workspace where only the OpenEnv server is available via ``ENV_BASE_URL``.

STDOUT FORMAT (mandatory — do not change):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Environment variables:
  HF_TOKEN              Hugging Face API token (or API_KEY / OPENAI_API_KEY)
  API_BASE_URL          LLM endpoint      (default: HF router)
  MODEL_NAME            Model id          (default: Qwen/Qwen2.5-7B-Instruct)
  WAREHOUSE_TASK        Optional: easy, medium, hard only
  ENV_BASE_URL          OpenEnv server    (default: http://127.0.0.1:7860)
  BENCHMARK             Logged env name   (default: warehouse-slotting-optimizer)
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_DIR = Path(__file__).resolve().parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from dotenv import load_dotenv
from openai import OpenAI

from client import WarehouseEnv
from grader import compute_episode_score
from models import WarehouseAction, WarehouseObservation
from tasks import get_task

TASK_ALIAS: Dict[str, str] = {
    "easy": "single_aisle_rebalance",
    "medium": "cross_aisle_constrained",
    "hard": "seasonal_changeover",
}

ALL_TASKS = ("easy", "medium", "hard")


def _load_env_files() -> None:
    for path in (_PROJECT_DIR / ".env", Path.cwd() / ".env"):
        if path.is_file():
            load_dotenv(path, encoding="utf-8-sig", override=True)


def _resolve_hf_token() -> str:
    _load_env_files()
    raw = (os.environ.get("HF_TOKEN") or "").strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        raw = raw[1:-1].strip()
    return raw


_load_env_files()

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860").rstrip("/")
BENCHMARK = os.getenv("BENCHMARK", "warehouse-slotting-optimizer")

TEMPERATURE = 0.2
MAX_TOKENS = 60
SUCCESS_THRESHOLD = 0.5


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    err_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP]  step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={err_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a warehouse slotting planner. Reply with ONE JSON object only (no markdown),
    matching this schema:
    {"action_type": "<inspect_zone|query_constraint|move_sku|swap_skus|freeze_zone|submit_plan>",
     "zone_id": "<optional string>",
     "sku_id": "<optional string>",
     "sku_to_move": "<optional string>",
     "target_zone": "<optional string>",
     "sku_a": "<optional string>",
     "sku_b": "<optional string>"}

    Goal: reduce weighted walk (pick_frequency * distance_to_dock) by moving high-frequency
    SKUs to zones with smaller distance_to_dock. Respect oversize (ground only), capacity,
    and do not put hazardous and food SKUs in the same aisle. Use empty zones for moves.
    You may inspect_zone or query_constraint before moving. Use submit_plan when done.
    """
).strip()


def build_user_prompt(obs: WarehouseObservation) -> str:
    lines: List[str] = [
        f"task={obs.task_name} step={obs.step_number} move_budget={obs.move_budget}",
        f"walk_distance={obs.current_walk_distance} baseline={obs.baseline_walk_distance}",
        f"pending_skus={obs.pending_skus}",
    ]
    if obs.last_action_error:
        lines.append(f"last_error={obs.last_action_error}")
    lines.append("zones: id -> aisle,level,cap_kg,dock_m,sku_in_slot,frozen")
    for zid in sorted(obs.zones.keys())[:100]:
        z = obs.zones[zid]
        lines.append(
            f"  {zid}: {z.aisle},{z.level},{z.capacity_kg},{z.distance_to_dock},"
            f"{z.current_sku},{z.is_frozen}"
        )
    lines.append("skus (id -> zone, kg, freq, haz, food, oversize):")
    for i, (sid, s) in enumerate(obs.skus.items()):
        if i >= 50:
            lines.append(f"  ... ({len(obs.skus) - 50} more)")
            break
        lines.append(
            f"  {sid}: {s.current_zone},{s.weight_kg},{s.pick_frequency},"
            f"{s.is_hazardous},{s.is_food},{s.is_oversize}"
        )
    if obs.constraint_violations:
        lines.append(f"violations_now={len(obs.constraint_violations)}")
    return "\n".join(lines)


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("No JSON object in model output")
    return json.loads(m.group(0))


def get_llm_action_json(client: OpenAI, obs: WarehouseObservation) -> str:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(obs)},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    raw = (resp.choices[0].message.content or "").strip()
    return " ".join(raw.split())


def parse_action(output: str) -> WarehouseAction:
    data = _extract_json_object(output)
    return WarehouseAction.model_validate(data)


def _episode_score(
    env_sync: Any,
    last_obs: WarehouseObservation,
    task_internal: str,
    cfg_move_budget: int,
) -> float:
    gs = last_obs.env_context.get("grader_score")
    if gs is not None:
        return float(gs)
    st = env_sync.state()
    if st.last_grader_score is not None:
        return float(st.last_grader_score)
    moves_used = cfg_move_budget - st.move_budget_remaining
    return compute_episode_score(
        task_internal,
        zones={k: v.model_dump() for k, v in st.zones.items()},
        skus={k: v.model_dump() for k, v in st.skus.items()},
        baseline_distance=st.baseline_walk_distance,
        steps_used=st.step_count,
        move_budget_used=moves_used,
    )


def run_warehouse_task(client: OpenAI, task_name: str) -> None:
    all_rewards: List[float] = []
    total_steps = 0
    score = 0.0
    success = False
    internal_id = TASK_ALIAS[task_name]
    cfg = get_task(internal_id)

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        with WarehouseEnv(base_url=ENV_BASE_URL).sync() as env_sync:
            try:
                result = env_sync.reset(task_id=internal_id)
                obs = result.observation
            except Exception as exc:
                log_step(1, "null", 0.0, True, str(exc))
                log_end(False, 1, 0.0, [0.0])
                return

            try:
                done = result.done
                while not done and total_steps < cfg.max_steps:
                    total_steps += 1
                    error_msg: Optional[str] = None
                    action_str = "null"
                    reward = 0.0

                    try:
                        action_str = get_llm_action_json(client, obs)
                        if not action_str:
                            action_str = "null"
                            raise ValueError("Model returned empty action")

                        action = parse_action(action_str)
                        action_str = json.dumps(
                            action.model_dump(mode="json"),
                            separators=(",", ":"),
                        )
                        result = env_sync.step(action)
                        reward = float(result.reward or 0.0)
                        done = result.done
                        obs = result.observation
                        if obs.last_action_error:
                            error_msg = obs.last_action_error
                    except Exception as exc:
                        if action_str == "null":
                            action_str = "ERROR"
                        error_msg = str(exc)
                        reward = -1.0
                        done = False

                    all_rewards.append(float(reward))
                    log_step(total_steps, action_str, float(reward), done, error_msg)

                score = float(
                    _episode_score(env_sync, obs, internal_id, cfg.move_budget),
                )
                score = round(min(max(score, 0.0), 1.0), 3)
                success = score >= SUCCESS_THRESHOLD

            except Exception as exc:
                print(
                    f"[DEBUG] Warehouse task error ({task_name}): {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            finally:
                log_end(success, total_steps, score, all_rewards)

    except Exception as exc:
        log_step(1, "null", 0.0, True, str(exc))
        log_end(False, 1, 0.0, [0.0])


def _tasks_to_run() -> List[str]:
    single = os.getenv("WAREHOUSE_TASK", "").strip().lower()
    if single:
        if single not in ALL_TASKS:
            print(
                f"[DEBUG] Invalid WAREHOUSE_TASK={single!r}; "
                f"expected one of {ALL_TASKS}",
                file=sys.stderr,
                flush=True,
            )
            return list(ALL_TASKS)
        return [single]
    return list(ALL_TASKS)


def main() -> None:
    key = (
        _resolve_hf_token()
        or (os.getenv("API_KEY") or "").strip()
        or (os.getenv("OPENAI_API_KEY") or "").strip()
    )
    if not key:
        print(
            "[DEBUG] HF_TOKEN (or API_KEY / OPENAI_API_KEY) is missing after loading:\n"
            f"  - {_PROJECT_DIR / '.env'}\n"
            f"  - {Path.cwd() / '.env'}\n"
            "Set HF_TOKEN for Hugging Face router, or OPENAI_API_KEY for OpenAI.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)

    llm = OpenAI(base_url=API_BASE_URL, api_key=key)

    for task_name in _tasks_to_run():
        run_warehouse_task(llm, task_name)


if __name__ == "__main__":
    main()
