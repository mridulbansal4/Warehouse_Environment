#!/usr/bin/env python3
"""
inference.py — Warehouse Slotting OpenEnv Baseline
==================================================

Runs warehouse tasks (easy / medium / hard) locally (grid env under repo root) and emits
the structured log format expected by the hackathon evaluator.

STDOUT FORMAT (mandatory — do not change):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Environment variables:
  HF_TOKEN              Hugging Face API token (required; set in .env or shell)
  API_BASE_URL          LLM endpoint      (default: HF router)
  MODEL_NAME            Model id          (default: Qwen/Qwen2.5-7B-Instruct)
  WAREHOUSE_TASK        Optional: run one of easy, medium, hard only
  ENV_BASE_URL          Unused for local grid env (OpenEnv server uses a different entry)
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path
from typing import List, Optional

# Grid env, grader, and legacy grid models live in the repo root (parent of this folder).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv
from openai import OpenAI

from env import WarehouseEnv
from grader import compute_score
from models import Action, MoveAction, Observation, Slot, SwapAction

_PROJECT_DIR = Path(__file__).resolve().parent


def _load_env_files() -> None:
    for path in (_PROJECT_DIR / ".env", _REPO_ROOT / ".env", Path.cwd() / ".env"):
        if path.is_file():
            load_dotenv(path, encoding="utf-8-sig", override=True)


def _resolve_hf_token() -> str:
    _load_env_files()
    raw = (os.environ.get("HF_TOKEN") or "").strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        raw = raw[1:-1].strip()
    return raw


_load_env_files()

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
BENCHMARK = os.getenv("BENCHMARK", "warehouse-slotting-optimizer")

TEMPERATURE = 0.2
MAX_TOKENS = 60
SUCCESS_THRESHOLD = 0.5

ALL_TASKS = ("easy", "medium", "hard")


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
    You control a warehouse slotting optimizer. Reply with EXACTLY ONE line:
    either MOVE or SWAP as specified below. No explanation, no markdown.

    MOVE SKU_ID row col level
    SWAP r1 c1 l1 r2 c2 l2

    Use only SKU IDs that appear in the current layout.
    """
).strip()


def build_user_prompt(state: Observation) -> str:
    return textwrap.dedent(
        f"""
        Current layout (slot -> SKU, null = empty):
        {state.layout}

        Rules (heuristics):
        - Higher velocity SKUs → closer to dispatch (0, 0, 0)
        - Heavy items (>15) → level 0 only when constraints are on
        - Chemical SKUs must not be adjacent to Food SKUs (when constraints are on)

        Respond with ONE action line only.
        """
    ).strip()


def parse_action(output: str) -> Action:
    parts = output.strip().split()
    if not parts:
        raise ValueError("Empty action")

    cmd = parts[0].upper()
    if cmd == "MOVE":
        if len(parts) < 5:
            raise ValueError("MOVE requires: MOVE SKU_ID row col level")
        return Action(
            action=MoveAction(
                type="move",
                sku_id=parts[1],
                target_slot=Slot(
                    row=int(parts[2]),
                    col=int(parts[3]),
                    level=int(parts[4]),
                ),
            )
        )

    if cmd == "SWAP":
        if len(parts) < 7:
            raise ValueError("SWAP requires 6 integers after SWAP")
        return Action(
            action=SwapAction(
                type="swap",
                slot_a=Slot(row=int(parts[1]), col=int(parts[2]), level=int(parts[3])),
                slot_b=Slot(row=int(parts[4]), col=int(parts[5]), level=int(parts[6])),
            )
        )

    raise ValueError(f"Unknown action: {parts[0]}")


def get_llm_action_line(client: OpenAI, state: Observation) -> str:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(state)},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    raw = (resp.choices[0].message.content or "").strip()
    return " ".join(raw.split())


def run_warehouse_task(client: OpenAI, task_name: str) -> None:
    all_rewards: List[float] = []
    total_steps = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        env = WarehouseEnv(task_name)
        state = env.reset()
    except Exception as exc:
        log_step(1, "null", 0.0, True, str(exc))
        log_end(False, 1, 0.0, [0.0])
        return

    try:
        done = False
        while not done and total_steps < env.max_steps:
            total_steps += 1
            error_msg: Optional[str] = None
            action_str = "null"
            reward = 0.0

            try:
                action_str = get_llm_action_line(client, state)
                if not action_str:
                    action_str = "null"
                    raise ValueError("Model returned empty action")

                action = parse_action(action_str)
                state, reward, done, info = env.step(action)
                if info.get("error"):
                    error_msg = info["error"]
            except Exception as exc:
                if action_str == "null":
                    action_str = "ERROR"
                error_msg = str(exc)
                reward = -1.0
                done = False

            all_rewards.append(float(reward))
            log_step(total_steps, action_str, float(reward), done, error_msg)

        score = float(compute_score(env))
        score = round(min(max(score, 0.0), 1.0), 3)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Warehouse task error ({task_name}): {exc}", file=sys.stderr, flush=True)
    finally:
        log_end(success, total_steps, score, all_rewards)


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
    hf_token = _resolve_hf_token()
    if not hf_token:
        print(
            "[DEBUG] HF_TOKEN is missing or empty after loading:\n"
            f"  - {_PROJECT_DIR / '.env'}\n"
            f"  - {_REPO_ROOT / '.env'}\n"
            f"  - {Path.cwd() / '.env'}\n"
            "Add a single line: HF_TOKEN=<your token>\n"
            "Or export HF_TOKEN in your shell (no quotes needed).",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=hf_token)

    for task_name in _tasks_to_run():
        run_warehouse_task(client, task_name)


if __name__ == "__main__":
    main()
