"""Context control for the local workforce model (8 GB VRAM budget).

Three levers keep long R&D runs inside a small, fixed context window:

- ``num_ctx()``: the explicit Ollama context length sent with every request
  (``OLLAMA_NUM_CTX`` env var, default 16384). Without it Ollama uses its own
  default, which silently changes with server settings.
- ``prune_tool_output()``: tool output keeps its head and its tail. Failures
  (tracebacks, assertion messages) sit at the end, so a plain prefix cut loses
  the one line that matters.
- ``compact_messages()``: tool results older than the most recent few are
  reduced to one-line summaries, and old long assistant turns are shortened.
  The message structure (roles, tool_calls, tool_name) is preserved so the
  Ollama tool-calling protocol stays valid.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

DEFAULT_NUM_CTX = 16384
DEFAULT_HEAD_LINES = 20
DEFAULT_TAIL_LINES = 50
DEFAULT_MAX_CHARS = 4000
DEFAULT_KEEP_RECENT_TOOLS = 3
DEFAULT_SUMMARY_CHARS = 160
DEFAULT_ASSISTANT_CHARS = 300


def num_ctx() -> int:
    """Returns the context length to request from Ollama."""
    raw = os.environ.get("OLLAMA_NUM_CTX", "").strip()
    try:
        value = int(raw) if raw else DEFAULT_NUM_CTX
    except ValueError:
        return DEFAULT_NUM_CTX
    return max(2048, value)


def ollama_timeout_s() -> float:
    """HTTP read timeout for one model call (``OLLAMA_TIMEOUT_S``, default 900).

    Must cover a full response at the output cap: 8192 tokens at ~20 tok/s is
    about 400 s, so a shorter timeout kills valid generations mid-answer.
    """
    raw = os.environ.get("OLLAMA_TIMEOUT_S", "").strip()
    try:
        value = float(raw) if raw else 900.0
    except ValueError:
        return 900.0
    return max(60.0, value)


def chat_num_predict() -> int:
    """Output-token cap for each chat turn (``OLLAMA_NUM_PREDICT``, default 8192).

    Without a cap, a model that starts repeating itself generates until the HTTP
    read timeout fires, holding the GPU and stalling the subtask.
    """
    raw = os.environ.get("OLLAMA_NUM_PREDICT", "").strip()
    try:
        value = int(raw) if raw else 8192
    except ValueError:
        return 8192
    return max(256, value)


def prune_tool_output(
    text: str,
    head: int = DEFAULT_HEAD_LINES,
    tail: int = DEFAULT_TAIL_LINES,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """Keeps the first ``head`` and last ``tail`` lines of long output.

    Short output passes through unchanged. The result never exceeds roughly
    ``max_chars``; if a kept slice is still too long, each side is trimmed.
    """
    text = "" if text is None else str(text)
    lines = text.splitlines()
    if len(lines) <= head + tail and len(text) <= max_chars:
        return text

    if len(lines) > head + tail:
        omitted = len(lines) - head - tail
        kept_head = lines[:head]
        kept_tail = lines[-tail:] if tail > 0 else []
        pruned = "\n".join(kept_head) + f"\n... [{omitted} lines omitted] ...\n" + "\n".join(kept_tail)
    else:
        pruned = text

    if len(pruned) > max_chars:
        half = max_chars // 2
        pruned = pruned[:half] + "\n... [truncated] ...\n" + pruned[-half:]
    return pruned


def _short(text: str, limit: int) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def compact_messages(
    messages: List[Dict[str, Any]],
    keep_recent_tools: int = DEFAULT_KEEP_RECENT_TOOLS,
    summary_chars: int = DEFAULT_SUMMARY_CHARS,
    assistant_chars: int = DEFAULT_ASSISTANT_CHARS,
    max_arg_chars: int = DEFAULT_MAX_CHARS,
    max_bytes: int | None = None,
    reserved_tokens: int | None = None,
) -> int:
    """Compacts old history in place and returns how many messages changed.

    The system prompt and the first user message (the task) are never touched.
    Recent tool results retain their contents while they fit. Older results are
    summarized, then entire exchanges are evicted to meet the UTF-8 byte budget.
    System messages and the original task are preserved; an irreducible prompt
    raises ValueError before inference.
    Already-executed ``tool_calls`` arguments exceeding ``max_arg_chars`` are also
    compacted immediately so a single huge ``write_file`` payload cannot trigger
    CUDA OOM on subsequent turns.
    """
    # Conservative UTF-8 byte estimate (one byte per token) leaves room for output.
    reserve = min(chat_num_predict(), num_ctx() // 2) if reserved_tokens is None else reserved_tokens
    limit = max_bytes if max_bytes is not None else max(1, num_ctx() - reserve)
    protected = {i for i, m in enumerate(messages) if m.get("role") == "system"}
    first_user = next((i for i, m in enumerate(messages) if m.get("role") == "user"), None)
    if first_user is not None:
        protected.add(first_user)

    def size(items):
        return len(json.dumps(items, ensure_ascii=False).encode("utf-8"))

    if size([m for i, m in enumerate(messages) if i in protected]) > limit:
        raise ValueError("Required system/task prompt exceeds context budget")

    def shorten_arguments(value):
        if isinstance(value, str):
            return f"[compacted {len(value)} chars] {_short(value, assistant_chars)}" if len(value) > max_arg_chars else value
        if isinstance(value, dict):
            return {k: shorten_arguments(v) for k, v in value.items()}
        if isinstance(value, list):
            if size(value) > max_arg_chars:
                return [f"[compacted {len(value)} argument items]"]
            return [shorten_arguments(v) for v in value]
        return value

    def enforce_budget():
        nonlocal changed
        # Delete entire old turns, never leave a tool response without its call.
        while size(messages) > limit:
            removable = next((i for i in range(len(messages)) if i not in protected), None)
            if removable is None:
                raise ValueError("Required prompt exceeds context budget")
            end = removable + 1
            while end < len(messages) and messages[end].get("role") == "tool":
                end += 1
            if end == len(messages):
                # Preserve the newest complete exchange; compress its evidence if needed.
                for msg in messages[removable:end]:
                    if "content" in msg:
                        msg["content"] = _short(msg.get("content", ""), summary_chars)
                    if msg.get("tool_calls"):
                        for call in msg["tool_calls"]:
                            fn = call.get("function", {})
                            if size(fn.get("arguments", {})) > summary_chars:
                                fn["arguments"] = {"_summary": "[compacted executed arguments]"}
                if size(messages) > limit:
                    raise ValueError("Newest complete tool exchange exceeds context budget")
                changed += 1
                break
            del messages[removable:end]
            shift = end - removable
            protected_copy = {i if i < removable else i - shift for i in protected}
            protected.clear()
            protected.update(protected_copy)
            changed += shift
    changed = 0

    for i, msg in enumerate(messages):
        if i in protected or msg.get("role") != "assistant":
            continue
        tcalls = msg.get("tool_calls")
        if not isinstance(tcalls, list):
            continue
        msg_changed = False
        for tc in tcalls:
            fn = tc.get("function") if isinstance(tc, dict) else None
            args = fn.get("arguments") if isinstance(fn, dict) else None
            if isinstance(fn, dict):
                compacted_args = shorten_arguments(args)
                if compacted_args != args:
                    fn["arguments"] = compacted_args
                    msg_changed = True
            if not isinstance(args, dict):
                continue
            for k, v in list(args.items()):
                if isinstance(v, str) and len(v) > max_arg_chars:
                    args[k] = f"[compacted {len(v)} chars] {_short(v, assistant_chars)}"
                    msg_changed = True
        if msg_changed:
            changed += 1

    tool_idx = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if len(tool_idx) <= keep_recent_tools:
        enforce_budget()
        return changed

    # First result inside the recent evidence window.
    boundary = tool_idx[-keep_recent_tools] if keep_recent_tools > 0 else len(messages)

    for i in tool_idx:
        if i >= boundary or i in protected:
            continue
        original = str(messages[i].get("content", ""))
        if len(original) <= summary_chars:
            continue
        name = messages[i].get("tool_name", "tool")
        messages[i]["content"] = f"[compacted {name}] {_short(original, summary_chars)}"
        changed += 1

    for i in range(boundary):
        if i in protected or messages[i].get("role") != "assistant":
            continue
        content = str(messages[i].get("content", "") or "")
        msg_changed = False
        if len(content) > assistant_chars:
            messages[i]["content"] = _short(content, assistant_chars)
            msg_changed = True
        tcalls = messages[i].get("tool_calls")
        if isinstance(tcalls, list):
            for tc in tcalls:
                fn = tc.get("function") if isinstance(tc, dict) else None
                args = fn.get("arguments") if isinstance(fn, dict) else None
                if not isinstance(args, dict):
                    continue
                for k, v in list(args.items()):
                    if isinstance(v, str) and len(v) > assistant_chars:
                        args[k] = _short(v, assistant_chars)
                        msg_changed = True
        if msg_changed:
            changed += 1

    enforce_budget()
    return changed
