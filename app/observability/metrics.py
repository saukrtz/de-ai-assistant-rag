"""
app/observability/metrics.py
─────────────────────────────
Lightweight in-process metrics stub.
No external Prometheus server required.
"""
from __future__ import annotations
import threading

_lock = threading.Lock()
_state: dict = {
    "metrics_server_initialized": False,
    "request_count": 0,
    "quality_action_count": 0,
}


def start_metrics_server(port: int = 8001) -> None:
    """Mark metrics as initialized (stub — no actual HTTP server needed)."""
    with _lock:
        _state["metrics_server_initialized"] = True


def get_metrics_status() -> dict:
    """Return current metrics snapshot."""
    with _lock:
        return dict(_state)


def increment_request_count() -> None:
    with _lock:
        _state["request_count"] += 1


def increment_quality_action_count() -> None:
    with _lock:
        _state["quality_action_count"] += 1
