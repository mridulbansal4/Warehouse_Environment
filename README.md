<<<<<<< HEAD
---
title: Warehouse Slotting OpenEnv
emoji: 📦
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Warehouse slotting (OpenEnv)

Realistic **warehouse slotting** tasks: place SKUs in zones to minimise weighted walking distance (pick frequency × distance to dock) under **capacity**, **oversize (ground-level)**, and **hazardous vs food aisle** rules. The server exposes the standard OpenEnv HTTP + WebSocket API (`reset`, `step`, `state`, `/schema`, `/docs`).

## Motivation

Slotting and re-slotting are everyday operations in fulfilment centres. This environment turns that into a structured decision process with **dense step rewards** (distance improvements, step cost, constraint penalties, bonuses) and a separate **deterministic episode grader** in `[0, 1]`.

## Action space (`WarehouseAction`)

All actions are one Pydantic model (`openenv` `Action` subclass):

| `action_type` | Fields | Effect |
|---------------|--------|--------|
| `inspect_zone` | `zone_id` | View a zone (redundant inspects penalised). |
| `query_constraint` | `sku_id` | Query SKU metadata (tracks first query per SKU). |
| `move_sku` | `sku_to_move`, `target_zone` | Move SKU to an **empty** zone (consumes move budget). |
| `swap_skus` | `sku_a`, `sku_b` | Swap two placed SKUs (consumes move budget). |
| `freeze_zone` | `zone_id` | Lock a zone against incoming moves. |
| `submit_plan` | — | End episode; applies final bonus from walk reduction vs target. |

Optional `metadata` on the base `Action` is allowed by OpenEnv.

## Observation space (`WarehouseObservation`)

Extends OpenEnv `Observation` (`done`, `reward`, `metadata`) plus:

- `env_context` — **Diagnostics visible over HTTP/WS** (OpenEnv strips `metadata` on the wire). Includes `reward_breakdown`, `violations`, walk distances, and `grader_score` when the episode ends.
- `zones`, `skus` — Full layout and SKU attributes.
- `task_name`, `pending_skus`, `move_budget`, walk distances, `constraint_violations`, `move_history`, `step_number`, `last_action_error`.

## Tasks and graders (easy → hard)

| Task ID | Difficulty | Summary |
|---------|------------|---------|
| `single_aisle_rebalance` | Easy | 5 SKUs, 8 zones, one aisle. |
| `cross_aisle_constrained` | Medium | 8 SKUs, 24 zones, 3 aisles, oversize + haz/food rules. |
| `seasonal_changeover` | Hard | 40 SKUs, 48 zones, seasonal inbound + large move budget. |

Each task ships a **deterministic** `TaskConfig.score(...)` in `tasks.py` (distance vs target, constraint violations, step efficiency). The environment writes the result to `observation.env_context["grader_score"]` when the episode finishes. The same logic is available as `compute_episode_score(...)` in `grader.py` for offline scoring from snapshots.

## Reward design

- **Per step:** small negative step cost; distance improvement vs baseline (clamped); bonus for acting soon after inspecting a zone; penalties for constraint violations and redundant inspects.
- **Submit / horizon:** final bonus if walk reduction meets (or partially meets) the task target; episode ends on `submit_plan`, exhausted move budget, or `max_steps`.

## Setup

```bash
cd Warehouse_env
uv sync
```

## Run the server (local)

```bash
uv run server
# or
uv run python -m Warehouse_env.server.app
```

Docker (from this directory, as on Hugging Face):

```bash
docker build -t warehouse-openenv:latest -f server/Dockerfile .
docker run --rm -p 8000:8000 warehouse-openenv:latest
```

## Client usage

```python
from Warehouse_env import WarehouseAction, WarehouseEnv, ActionType

with WarehouseEnv(base_url="http://127.0.0.1:8000").sync() as env:
    r = env.reset(task_id="single_aisle_rebalance")
    r = env.step(
        WarehouseAction(
            action_type=ActionType.MOVE_SKU,
            sku_to_move="SKU_001",
            target_zone="A1",
        )
    )
    s = env.state()
```

`reset(...)` accepts OpenEnv parameters (`seed`, `episode_id`) plus **`task_id`** to select a task.

## Baseline inference (evaluator log format)

Requires a **running OpenEnv server**. Uses the same **stdout format** as the root hackathon baseline:

`[START]`, `[STEP]`, `[END]` lines with `task=easy|medium|hard`, `env=BENCHMARK`, etc.

Environment variables (aligned with the parent `inference.py`):

- **`HF_TOKEN`** — Hugging Face token for `API_BASE_URL` (default HF router), or use **`API_KEY`** / **`OPENAI_API_KEY`** for other endpoints.
- **`API_BASE_URL`** — chat completions base URL (default `https://router.huggingface.co/v1`).
- **`MODEL_NAME`** — model id (default `Qwen/Qwen2.5-7B-Instruct`).
- **`ENV_BASE_URL`** — OpenEnv server (default `http://127.0.0.1:8000`).
- **`WAREHOUSE_TASK`** — optional: `easy`, `medium`, or `hard` (maps to internal task ids); default = all three.
- **`BENCHMARK`** — name in `[START]` line (default `warehouse-slotting-optimizer`).

```bash
export HF_TOKEN=...
export ENV_BASE_URL=http://127.0.0.1:8000   # optional

uv run warehouse-baseline
# or: uv run python -m Warehouse_env.inference
```

The **`[END] score=`** value is the deterministic episode grader in `[0,1]` (`grader_score` / `TaskConfig.score`).

## Validation

```bash
uv run openenv validate
```

## Baseline scores (reference)

Scores depend on model and sampling. Example placeholder after a local run with `gpt-4o-mini` (fill in your own runs in submissions):

| Task | Typical grader_score (informal) |
|------|----------------------------------|
| single_aisle_rebalance | model-dependent |
| cross_aisle_constrained | model-dependent |
| seasonal_changeover | model-dependent |

Re-run `warehouse-baseline` and paste the printed `grader_score` lines for your README when publishing.

## Layout

```
Warehouse_env/
├── openenv.yaml
├── pyproject.toml
├── uv.lock
├── README.md
├── __init__.py
├── models.py          # Action / Observation / State + domain types
├── tasks.py           # Scenarios + TaskConfig.score
├── grader.py          # Snapshot scoring helper
├── client.py          # WebSocket EnvClient
├── inference.py       # OpenAI baseline driver
└── server/
    ├── app.py         # FastAPI app (create_app)
    ├── Dockerfile
    ├── requirements.txt
    └── Warehouse_env_environment.py
```

## Deploying to Hugging Face Spaces

Use **`sdk: docker`** (see README front matter) and tag the Space with **`openenv`**. From this directory:

```bash
openenv push
```

(Requires Hugging Face CLI login when prompted.)
=======

