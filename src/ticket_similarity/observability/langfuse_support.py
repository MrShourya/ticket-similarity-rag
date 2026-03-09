import os
from typing import Any, Callable

from dotenv import load_dotenv

# Load .env once
load_dotenv()

_LANGFUSE_CLIENT = None
_LANGFUSE_ACTIVE = None


def _is_enabled() -> bool:
    return os.getenv("LANGFUSE_ENABLED", "false").strip().lower() == "true"


def _has_minimum_config() -> bool:
    return all(
        [
            os.getenv("LANGFUSE_PUBLIC_KEY"),
            os.getenv("LANGFUSE_SECRET_KEY"),
            os.getenv("LANGFUSE_BASE_URL"),
        ]
    )


def is_langfuse_active() -> bool:
    global _LANGFUSE_ACTIVE

    if _LANGFUSE_ACTIVE is not None:
        return _LANGFUSE_ACTIVE

    _LANGFUSE_ACTIVE = _is_enabled() and _has_minimum_config()
    return _LANGFUSE_ACTIVE


def get_langfuse_client():
    """
    Lazily initialize and cache the Langfuse client.
    Returns None if disabled or initialization fails.
    """
    global _LANGFUSE_CLIENT

    if not is_langfuse_active():
        return None

    if _LANGFUSE_CLIENT is not None:
        return _LANGFUSE_CLIENT

    try:
        from langfuse import get_client

        _LANGFUSE_CLIENT = get_client()
        return _LANGFUSE_CLIENT
    except Exception as e:
        print(f"[Langfuse] client initialization failed: {e}")
        return None


def observe(name: str | None = None):
    """
    Safe decorator:
    - no-op if Langfuse is disabled
    - uses Langfuse observe() if enabled
    """
    def decorator(func: Callable):
        if not is_langfuse_active():
            return func

        try:
            from langfuse import observe as lf_observe
            return lf_observe(name=name)(func)
        except Exception as e:
            print(f"[Langfuse] observe decorator fallback for {func.__name__}: {e}")
            return func

    return decorator


def update_current_trace(
    name: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
    input: Any = None,
    output: Any = None,
    metadata: dict | None = None,
) -> None:
    """
    Best-effort trace update.
    Works when the installed Langfuse client exposes update_current_trace().
    Otherwise becomes a silent no-op.
    """
    client = get_langfuse_client()
    if client is None:
        return

    payload = {}
    if name is not None:
        payload["name"] = name
    if user_id is not None:
        payload["user_id"] = user_id
    if session_id is not None:
        payload["session_id"] = session_id
    if tags is not None:
        payload["tags"] = tags
    if input is not None:
        payload["input"] = input
    if output is not None:
        payload["output"] = output
    if metadata is not None:
        payload["metadata"] = metadata

    try:
        update_fn = getattr(client, "update_current_trace", None)
        if callable(update_fn):
            update_fn(**payload)
            return
    except Exception as e:
        print(f"[Langfuse] update_current_trace failed: {e}")


def update_current_observation(
    input: Any = None,
    output: Any = None,
    metadata: dict | None = None,
    level: str | None = None,
    status_message: str | None = None,
    tags: list[str] | None = None,
) -> None:
    """
    Best-effort observation update.
    Some Langfuse SDK versions expose update_current_observation(),
    others may not. If unavailable, this safely becomes a no-op.
    """
    client = get_langfuse_client()
    if client is None:
        return

    payload = {}
    if input is not None:
        payload["input"] = input
    if output is not None:
        payload["output"] = output
    if metadata is not None:
        payload["metadata"] = metadata
    if level is not None:
        payload["level"] = level
    if status_message is not None:
        payload["status_message"] = status_message
    if tags is not None:
        payload["tags"] = tags

    try:
        update_fn = getattr(client, "update_current_observation", None)
        if callable(update_fn):
            update_fn(**payload)
            return

        # No method on this SDK version -> safe no-op
    except Exception as e:
        print(f"[Langfuse] update_current_observation failed: {e}")


def flush_langfuse() -> None:
    client = get_langfuse_client()
    if client is None:
        return

    try:
        client.flush()
        print("[Langfuse] flush complete")
    except Exception as e:
        print(f"[Langfuse] flush failed: {e}")


def shutdown_langfuse() -> None:
    client = get_langfuse_client()
    if client is None:
        return

    try:
        client.shutdown()
        print("[Langfuse] shutdown complete")
    except Exception as e:
        print(f"[Langfuse] shutdown failed: {e}")