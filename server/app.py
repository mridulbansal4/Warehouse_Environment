# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Warehouse Env Environment.

This module creates an HTTP server that exposes the WarehouseEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app

    Swagger UI: open http://127.0.0.1:8000/docs (or http://localhost:8000/docs).
    Do not use http://0.0.0.0:8000 in the browser — 0.0.0.0 is only a bind address.
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import WarehouseAction, WarehouseObservation
    from .Warehouse_env_environment import WarehouseEnvironment
except (ImportError, ModuleNotFoundError):
    from models import WarehouseAction, WarehouseObservation
    from server.Warehouse_env_environment import WarehouseEnvironment


# Create the app with web interface and README integration
app = create_app(
    WarehouseEnvironment,
    WarehouseAction,
    WarehouseObservation,
    env_name="Warehouse_env",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)


def main(host: str = "0.0.0.0", port: int | None = None) -> None:
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        python -m Warehouse_env.server.app

    Port: argument ``port``, else env ``PORT``, else 8000.

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn Warehouse_env.server.app:app --workers 4
    """
    import os

    import uvicorn

    listen_port = int(port) if port is not None else int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=listen_port)


if __name__ == "__main__":
    main()
