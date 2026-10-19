"""Ollama HTTP streaming client for on-device Gemma 4 12B model."""

import asyncio
import contextlib
import json
import os
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

import urllib.error
import urllib.request

from antigravity_hybrid_orchestrator.workforce.context import (
    bytes_per_token,
    chat_num_predict,
    compact_messages,
    effective_num_predict,
    num_ctx,
    ollama_think,
    ollama_timeout_s,
    preflight_context_budget,
    tool_call_reserve,
)


class OllamaConnectionError(Exception):
    """Raised when connection to local Ollama service fails."""
    pass


class WorkforceClient:
    """Client for streaming token generation from local Ollama service."""

    DEFAULT_BASE_URL = "http://127.0.0.1:11434"
    DEFAULT_MODEL = "gemma4:12b"
    _model_digests: Dict[str, str] = {}

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        think: Optional[bool] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = (model or os.environ.get("OLLAMA_MODEL", "")).strip() or self.DEFAULT_MODEL
        self.timeout = (timeout if timeout > 0 else None) if timeout is not None else ollama_timeout_s()
        self.think = think

    def _effective_think(self, override: Optional[bool] = None) -> bool:
        if override is not None:
            return bool(override)
        if self.think is not None:
            return bool(self.think)
        return ollama_think()

    def get_model_digest(self) -> str:
        """Return the SHA-256 digest of ``self.model`` from Ollama ``/api/tags`` (cached)."""
        cache_key = f"{self.base_url}::{self.model}"
        if cache_key in self._model_digests:
            return self._model_digests[cache_key]
        url = f"{self.base_url}/api/tags"
        try:
            if HAS_HTTPX:
                with httpx.Client(timeout=3.0) as client:
                    resp = client.get(url)
                    if resp.status_code != 200:
                        return "unknown"
                    data = resp.json()
            else:
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    if resp.status != 200:
                        return "unknown"
                    data = json.loads(resp.read().decode("utf-8"))
            for item in data.get("models", []) or []:
                if item.get("name") == self.model or item.get("model") == self.model:
                    digest = str(item.get("digest", "") or "").strip()
                    if digest:
                        self._model_digests[cache_key] = digest
                        return digest
        except Exception:
            return "unknown"
        return "unknown"


    def check_health(self) -> bool:
        """Check if local Ollama daemon is reachable."""
        url = f"{self.base_url}/api/version"
        try:
            if HAS_HTTPX:
                with httpx.Client(timeout=10.0) as client:
                    resp = client.get(url)
                    return resp.status_code == 200
            else:
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=10.0) as resp:
                    return resp.status == 200
        except Exception:
            return False

    def warmup(self) -> bool:
        """Pre-warm or verify the model is loaded in Ollama to prevent cold-start latency."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": "ping",
            "stream": False,
            "options": {"num_predict": 1},
        }
        try:
            if HAS_HTTPX:
                with httpx.Client(timeout=60.0) as client:
                    resp = client.post(url, json=payload)
                    return resp.status_code == 200
            else:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=60.0) as resp:
                    return resp.status == 200
        except Exception:
            return False

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
    ) -> Generator[Tuple[str, str, Dict[str, Any]], None, None]:
        """Stream token generation from Ollama HTTP API.

        Yields:
            Tuple of (token_chunk, accumulated_text, metadata_dict).
            The final yield contains done=True and evaluation metrics.

        Raises:
            OllamaConnectionError: If connection fails or stream is interrupted.
        """
        url = f"{self.base_url}/api/generate"
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "think": False,
            "options": {
                "temperature": temperature,
                "num_predict": 2048,
                "num_ctx": num_ctx(),
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        accumulated_text = ""

        if HAS_HTTPX:
            # Use robust httpx streaming client with separate connect/read timeouts
            client_timeout = httpx.Timeout(
                timeout=self.timeout,
                connect=60.0,
                read=self.timeout,
                write=60.0,
                pool=60.0,
            )
            try:
                with httpx.Client(timeout=client_timeout) as client:
                    with client.stream("POST", url, json=payload) as resp:
                        if resp.status_code != 200:
                            raise OllamaConnectionError(
                                f"Ollama HTTP error {resp.status_code}: {resp.read().decode('utf-8')}"
                            )

                        for line in resp.iter_lines():
                            line = line.strip()
                            if not line:
                                continue

                            try:
                                chunk = json.loads(line)
                            except json.JSONDecodeError:
                                continue

                            token = chunk.get("response", "")
                            accumulated_text += token

                            is_done = chunk.get("done", False)
                            meta: Dict[str, Any] = {"done": is_done}

                            if is_done:
                                eval_count = int(chunk.get("eval_count", 0))
                                eval_duration_ns = int(chunk.get("eval_duration", 0))
                                eval_duration_s = (
                                    eval_duration_ns / 1_000_000_000.0 if eval_duration_ns > 0 else 0.0
                                )
                                prompt_eval_count = int(chunk.get("prompt_eval_count", 0))
                                total_duration_ns = int(chunk.get("total_duration", 0))
                                total_duration_s = (
                                    total_duration_ns / 1_000_000_000.0 if total_duration_ns > 0 else 0.0
                                )

                                meta.update({
                                    "eval_count": eval_count,
                                    "eval_duration_s": eval_duration_s,
                                    "prompt_eval_count": prompt_eval_count,
                                    "total_duration_s": total_duration_s,
                                })

                            yield token, accumulated_text, meta

            except httpx.ConnectError as e:
                raise OllamaConnectionError(f"Failed to connect to Ollama at {url}: {e}") from e
            except httpx.TimeoutException as e:
                raise OllamaConnectionError(f"Ollama streaming generation timed out after {self.timeout}s: {e}") from e
            except Exception as e:
                raise OllamaConnectionError(f"Ollama streaming generation error: {e}") from e

        else:
            # Fallback to urllib
            data_bytes = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data_bytes,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    for raw_line in resp:
                        line = raw_line.decode("utf-8").strip()
                        if not line:
                            continue

                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        token = chunk.get("response", "")
                        accumulated_text += token

                        is_done = chunk.get("done", False)
                        meta: Dict[str, Any] = {"done": is_done}

                        if is_done:
                            eval_count = int(chunk.get("eval_count", 0))
                            eval_duration_ns = int(chunk.get("eval_duration", 0))
                            eval_duration_s = (
                                eval_duration_ns / 1_000_000_000.0 if eval_duration_ns > 0 else 0.0
                            )
                            prompt_eval_count = int(chunk.get("prompt_eval_count", 0))
                            total_duration_ns = int(chunk.get("total_duration", 0))
                            total_duration_s = (
                                total_duration_ns / 1_000_000_000.0 if total_duration_ns > 0 else 0.0
                            )

                            meta.update({
                                "eval_count": eval_count,
                                "eval_duration_s": eval_duration_s,
                                "prompt_eval_count": prompt_eval_count,
                                "total_duration_s": total_duration_s,
                            })

                        yield token, accumulated_text, meta

            except urllib.error.URLError as e:
                raise OllamaConnectionError(f"Failed to connect to Ollama at {url}: {e}") from e
            except Exception as e:
                raise OllamaConnectionError(f"Ollama streaming generation error: {e}") from e

    async def _guarded_post(self, url, payload, deadline, execution_control):
        # An asyncio cancellation interrupts the socket operation itself: no abandoned
        # background inference thread survives a timeout or Ctrl+C.
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            pending = asyncio.create_task(client.post(url, json=payload))
            try:
                while not pending.done():
                    if deadline is not None and time.monotonic() >= deadline:
                        raise OllamaConnectionError("Ollama call deadline exceeded")
                    if execution_control is not None:
                        stop = execution_control.check()
                        if stop.stopped:
                            raise OllamaConnectionError(f"Ollama call stopped: {stop.reason}")
                    wait = .05 if deadline is None else max(.001, min(.05, deadline - time.monotonic()))
                    await asyncio.wait({pending}, timeout=wait)
                response = await pending
                response.raise_for_status()
                return response.json()
            finally:
                if not pending.done():
                    pending.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await pending

    def _post_json(self, url: str, payload: Dict[str, Any], *, deadline=None, execution_control=None) -> Dict[str, Any]:
        """POST JSON, enforcing a whole-call deadline when execution is bounded."""
        if deadline is not None or execution_control is not None:
            if not HAS_HTTPX:
                raise OllamaConnectionError("httpx is required for cancellable model calls")
            try:
                return asyncio.run(self._guarded_post(url, payload, deadline, execution_control))
            except OllamaConnectionError:
                raise
            except Exception as exc:
                raise OllamaConnectionError(f"Ollama request error: {exc}") from exc
        if HAS_HTTPX:
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(url, json=payload)
                    if resp.status_code != 200:
                        raise OllamaConnectionError(
                            f"Ollama HTTP error {resp.status_code}: {resp.text}"
                        )
                    return resp.json()
            except httpx.ConnectError as e:
                raise OllamaConnectionError(f"Failed to connect to Ollama at {url}: {e}") from e
            except httpx.TimeoutException as e:
                raise OllamaConnectionError(f"Ollama request timed out after {self.timeout}s: {e}") from e
            except OllamaConnectionError:
                raise
            except Exception as e:
                raise OllamaConnectionError(f"Ollama request error: {e}") from e
        else:
            data_bytes = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data_bytes,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.URLError as e:
                raise OllamaConnectionError(f"Failed to connect to Ollama at {url}: {e}") from e
            except Exception as e:
                raise OllamaConnectionError(f"Ollama request error: {e}") from e

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
        *,
        think: Optional[bool] = None,
        seed: Optional[int] = None,
        response_format: Optional[Any] = None,
        deadline: Optional[float] = None,
        execution_control: Any = None,
    ) -> Dict[str, Any]:
        """Non-streaming `/api/chat` call with native Ollama tool-calling and Thinking support.

        Args:
            messages: Chat history (roles: system/user/assistant/tool).
            tools: Optional tool schemas (OpenAI-style function definitions).
            temperature: Sampling temperature.
            think: Optional per-call Thinking Mode override (defaults to ``OLLAMA_THINK``).
            seed: Optional deterministic RNG seed (defaults to ``OLLAMA_SEED`` if set).
            response_format: Optional JSON schema or ``"json"`` format constraint.

        Returns:
            Dict with ``content``, ``thinking``, ``tool_calls``, token/duration telemetry,
            ``done_reason``, ``truncated``, and budget diagnostics.
        """
        url = f"{self.base_url}/api/chat"
        ctx_limit = num_ctx()
        output_cap = effective_num_predict(has_tools=bool(tools), num_ctx_val=ctx_limit)
        think_flag = self._effective_think(think)
        effective_seed = seed
        if effective_seed is None:
            raw_seed = os.environ.get("OLLAMA_SEED", "").strip()
            if raw_seed:
                try:
                    effective_seed = int(raw_seed)
                except ValueError:
                    effective_seed = None
        options: Dict[str, Any] = {
            "temperature": temperature,
            "num_ctx": ctx_limit,
            "num_predict": output_cap,
        }
        if effective_seed is not None:
            options["seed"] = effective_seed
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": think_flag,
            "options": options,
        }
        if tools:
            payload["tools"] = tools
        if response_format is not None:
            payload["format"] = response_format

        # Budget tool schemas as well as history; convert available token budget to bytes
        # so token and byte units are properly aligned.
        preflight = preflight_context_budget(
            messages,
            tools=tools,
            num_ctx_val=ctx_limit,
            num_predict_val=output_cap,
            extra_payload={k: v for k, v in payload.items() if k not in ("messages", "tools")},
        )
        compacted_count = compact_messages(
            messages,
            max_bytes=preflight["max_message_bytes"],
            reserved_tokens=output_cap,
        )
        if deadline is not None or execution_control is not None:
            raw = self._post_json(url, payload, deadline=deadline, execution_control=execution_control)
        else:
            raw = self._post_json(url, payload)
        message = raw.get("message", {}) or {}
        content = message.get("content", "") or ""
        thinking = message.get("thinking", "") or ""
        done_reason = str(raw.get("done_reason", "") or "")

        tool_calls: List[Dict[str, Any]] = []
        malformed_tool_calls = 0
        for tc in message.get("tool_calls", []) or []:
            fn = tc.get("function", {}) or {}
            args = fn.get("arguments", {})
            malformed = False
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
                    malformed = True
                    malformed_tool_calls += 1
            elif not isinstance(args, dict):
                args = {}
                malformed = True
                malformed_tool_calls += 1
            call_entry: Dict[str, Any] = {"name": fn.get("name", ""), "arguments": args or {}}
            if malformed:
                call_entry["malformed_arguments"] = True
            tool_calls.append(call_entry)

        prompt_eval_count = int(raw.get("prompt_eval_count", 0))
        prompt_eval_cached_count = int(raw.get("prompt_eval_cached_count", 0))
        eval_count = int(raw.get("eval_count", 0))
        eval_duration_ns = int(raw.get("eval_duration", 0))
        prompt_eval_duration_ns = int(raw.get("prompt_eval_duration", 0))
        total_duration_ns = int(raw.get("total_duration", 0))

        truncated = (done_reason == "length") or (
            done_reason == "" and output_cap > 0 and eval_count >= output_cap and not tool_calls
        )

        return {
            "content": content,
            "thinking": thinking,
            "has_thinking": bool(thinking),
            "thinking_chars": len(thinking),
            "tool_calls": tool_calls,
            "malformed_tool_calls": malformed_tool_calls,
            "input_tokens": prompt_eval_count,
            "prompt_eval_cached_tokens": prompt_eval_cached_count,
            "output_tokens": eval_count,
            "done_reason": done_reason,
            "truncated": bool(truncated),
            "think_enabled": think_flag,
            "num_ctx": ctx_limit,
            "num_predict": output_cap,
            "input_budget_tokens": preflight["input_budget_tokens"],
            "compacted_messages": compacted_count,
            "eval_duration_s": eval_duration_ns / 1_000_000_000.0 if eval_duration_ns > 0 else 0.0,
            "prompt_eval_duration_s": (
                prompt_eval_duration_ns / 1_000_000_000.0 if prompt_eval_duration_ns > 0 else 0.0
            ),
            "total_duration_s": total_duration_ns / 1_000_000_000.0 if total_duration_ns > 0 else 0.0,
        }
