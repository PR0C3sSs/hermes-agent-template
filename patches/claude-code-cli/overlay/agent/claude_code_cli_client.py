"""OpenAI-compatible client facade for Claude Code CLI print mode.

This lets Hermes use a local, already-authenticated ``claude`` binary as a
primary model provider without calling Anthropic's third-party API path.  The
wire shape presented to Hermes is the OpenAI Chat Completions surface, while the
underlying execution mirrors OpenClaw's Claude CLI backend: ``claude -p`` with
``--output-format stream-json`` and the user settings source.
"""

from __future__ import annotations

import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

from agent.redact import redact_sensitive_text

CLAUDE_CODE_MARKER_BASE_URL = "claude-code-cli://claude"
_DEFAULT_TIMEOUT_SECONDS = 1800.0

# Keep Claude Code itself in pure-LLM mode. Hermes owns tools; Claude should
# express desired tool use as OpenAI-style function calls for Hermes to execute.
# The names here are valid in Claude Code 2.1.x; future unknown names only emit a
# CLI warning on stderr and do not fail the request.
_DEFAULT_DISALLOWED_TOOLS = (
    "Task",
    "Bash",
    "CronCreate",
    "CronDelete",
    "CronList",
    "DesignSync",
    "Edit",
    "EnterWorktree",
    "ExitWorktree",
    "Glob",
    "Grep",
    "Monitor",
    "NotebookEdit",
    "PushNotification",
    "Read",
    "RemoteTrigger",
    "ReportFindings",
    "ScheduleWakeup",
    "SendMessage",
    "Skill",
    "TaskCreate",
    "TaskGet",
    "TaskList",
    "TaskOutput",
    "TaskStop",
    "TaskUpdate",
    "ToolSearch",
    "WebFetch",
    "WebSearch",
    "Workflow",
    "Write",
)

_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL | re.IGNORECASE
)
_TOOL_CALL_JSON_RE = re.compile(
    r"\{\s*\"id\"\s*:\s*\"[^\"]+\"\s*,\s*\"type\"\s*:\s*\"function\"\s*,\s*\"function\"\s*:\s*\{.*?\}\s*\}",
    re.DOTALL,
)


def _resolve_command() -> str:
    return (
        os.getenv("HERMES_CLAUDE_CODE_COMMAND", "").strip()
        or os.getenv("CLAUDE_CODE_CLI_PATH", "").strip()
        or "claude"
    )


def _resolve_args() -> list[str]:
    raw = os.getenv("HERMES_CLAUDE_CODE_ARGS", "").strip()
    if raw:
        return shlex.split(raw)
    return [
        "-p",
        "--output-format",
        "stream-json",
        "--include-partial-messages",
        "--verbose",
        "--strict-mcp-config",
        "--setting-sources",
        "user",
        "--disallowedTools",
        ",".join(_DEFAULT_DISALLOWED_TOOLS),
    ]


def _resolve_home_dir() -> str:
    home = os.environ.get("HOME", "").strip()
    if home:
        return home
    expanded = os.path.expanduser("~")
    if expanded and expanded != "~":
        return expanded
    try:
        import pwd

        resolved = pwd.getpwuid(os.getuid()).pw_dir.strip()
        if resolved:
            return resolved
    except Exception:
        pass
    return "/tmp"


def _build_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["HOME"] = _resolve_home_dir()
    try:
        from hermes_constants import apply_subprocess_home_env

        apply_subprocess_home_env(env)
    except Exception:
        pass

    # Match OpenClaw's important safety behavior: do not let Anthropic API env
    # vars turn Claude Code into a third-party API call. Claude Code should use
    # its own persisted login under HOME/CLAUDE_CONFIG_DIR.
    for key in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_CUSTOM_HEADERS",
        "CLAUDE_CODE_OAUTH_TOKEN",
    ):
        env.pop(key, None)
    return env


def _render_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text") or "").strip()
        if "content" in content and isinstance(content.get("content"), str):
            return str(content.get("content") or "").strip()
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
                elif item.get("type") == "image_url":
                    parts.append("[image input omitted for Claude Code CLI provider]")
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(p for p in parts if p).strip()
    return str(content).strip()


def _format_tool_specs(tools: list[dict[str, Any]] | None) -> str:
    if not isinstance(tools, list) or not tools:
        return ""
    tool_specs: list[dict[str, Any]] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") or {}
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        tool_specs.append(
            {
                "name": name.strip(),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            }
        )
    if not tool_specs:
        return ""
    return json.dumps(tool_specs, ensure_ascii=False)


def _format_messages_as_prompt(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
) -> str:
    sections: list[str] = [
        "You are Claude Code CLI being used as Hermes Agent's active model provider.",
        "Do not use Claude Code's own tools or side-effect capabilities. Hermes owns all tool execution.",
        "If a tool/action is needed, emit exactly one or more tool calls using this XML form and no extra prose:",
        '<tool_call>{"id":"call_1","type":"function","function":{"name":"tool_name","arguments":"{\\"arg\\":\\"value\\"}"}}</tool_call>',
        "The function.arguments value MUST be a JSON string. If no tool is needed, answer normally.",
    ]
    if model:
        sections.append(f"Hermes requested Claude model: {model}")

    tool_specs = _format_tool_specs(tools)
    if tool_specs:
        sections.append("Available Hermes tools (OpenAI function schema):\n" + tool_specs)
    if tool_choice is not None:
        sections.append("Tool choice hint: " + json.dumps(tool_choice, ensure_ascii=False))

    transcript: list[str] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown").strip().lower()
        label = {
            "system": "System",
            "developer": "System",
            "user": "User",
            "assistant": "Assistant",
            "tool": "Tool result",
        }.get(role, role.title())

        rendered = _render_message_content(message.get("content"))
        if role == "assistant" and message.get("tool_calls"):
            try:
                rendered_calls = json.dumps(message.get("tool_calls"), ensure_ascii=False)
            except Exception:
                rendered_calls = str(message.get("tool_calls"))
            rendered = (rendered + "\n" if rendered else "") + "Tool calls requested:\n" + rendered_calls
        if role == "tool" and message.get("tool_call_id"):
            rendered = f"tool_call_id={message.get('tool_call_id')}\n{rendered}"
        if not rendered:
            continue
        transcript.append(f"{label}:\n{rendered}")

    if transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))
    sections.append("Continue from the latest user request.")
    return "\n\n".join(s.strip() for s in sections if s and s.strip())


def _extract_tool_calls_from_text(text: str) -> tuple[list[SimpleNamespace], str]:
    if not isinstance(text, str) or not text.strip():
        return [], ""
    extracted: list[SimpleNamespace] = []
    consumed_spans: list[tuple[int, int]] = []

    def _try_add(raw_json: str) -> None:
        try:
            obj = json.loads(raw_json)
        except Exception:
            return
        if not isinstance(obj, dict):
            return
        fn = obj.get("function")
        if not isinstance(fn, dict):
            return
        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            return
        arguments = fn.get("arguments", "{}")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        call_id = obj.get("id")
        if not isinstance(call_id, str) or not call_id.strip():
            call_id = f"claude_code_call_{len(extracted) + 1}"
        extracted.append(
            SimpleNamespace(
                id=call_id,
                call_id=call_id,
                response_item_id=None,
                type="function",
                function=SimpleNamespace(name=name.strip(), arguments=arguments),
            )
        )

    for match in _TOOL_CALL_BLOCK_RE.finditer(text):
        _try_add(match.group(1))
        consumed_spans.append((match.start(), match.end()))
    if not extracted:
        for match in _TOOL_CALL_JSON_RE.finditer(text):
            _try_add(match.group(0))
            consumed_spans.append((match.start(), match.end()))

    if not consumed_spans:
        return extracted, text.strip()
    consumed_spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in consumed_spans:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    parts: list[str] = []
    cursor = 0
    for start, end in merged:
        if cursor < start:
            parts.append(text[cursor:start])
        cursor = max(cursor, end)
    if cursor < len(text):
        parts.append(text[cursor:])
    cleaned = "\n".join(p.strip() for p in parts if p and p.strip()).strip()
    return extracted, cleaned


def _namespace_usage(raw: dict[str, Any] | None) -> SimpleNamespace:
    raw = raw or {}
    input_tokens = int(raw.get("input_tokens") or 0)
    output_tokens = int(raw.get("output_tokens") or 0)
    cache_creation = int(raw.get("cache_creation_input_tokens") or 0)
    cache_read = int(raw.get("cache_read_input_tokens") or 0)
    prompt_tokens = input_tokens + cache_creation + cache_read
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=output_tokens,
        total_tokens=prompt_tokens + output_tokens,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cache_read),
    )


def _has_flag(args: list[str], *flags: str) -> bool:
    """Return True when an argv list contains any flag.

    Accept both ``--flag value`` and ``--flag=value`` forms. Claude Code
    supports both for many options, and provider safety checks should not append
    duplicate flags just because a caller used the equals form in
    ``HERMES_CLAUDE_CODE_ARGS``.
    """
    return any(arg in flags or any(arg.startswith(f"{flag}=") for flag in flags) for arg in args)


def _is_model_placeholder(model: str | None) -> bool:
    m = (model or "").strip().lower()
    return not m or m in {"claude-code-cli", "claude-cli", "claude-code", "claude"}


def _extract_result_status(stdout: str) -> dict[str, Any]:
    """Return the final Claude Code ``type=result`` status, if present."""
    status: dict[str, Any] = {}
    for raw_line in (stdout or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict) and obj.get("type") == "result":
            status = obj
    return status


def _summarize_failure(stderr: str, stdout: str, returncode: int | None) -> str:
    """Build a concise, human-readable CLI failure summary.

    Claude Code stream-json stdout can be very large.  Dumping the raw JSON into
    Hermes' retry error both hides the useful message (e.g. ``Prompt is too
    long``) and leaks distracting usage/cost telemetry into Telegram.  Prefer
    the structured ``type=result`` fields when available, then fall back to
    stderr/stdout tails.
    """
    status = _extract_result_status(stdout)
    pieces: list[str] = []
    result = status.get("result")
    if isinstance(result, str) and result.strip():
        pieces.append(result.strip())
    errors = status.get("errors")
    if isinstance(errors, list):
        err_text = "; ".join(str(e) for e in errors if str(e).strip())
        if err_text and err_text not in pieces:
            pieces.append(err_text)
    terminal_reason = status.get("terminal_reason")
    if isinstance(terminal_reason, str) and terminal_reason.strip() and not pieces:
        pieces.append(f"terminal_reason={terminal_reason.strip()}")
    safe_stderr = redact_sensitive_text((stderr or "").strip()[-4000:], force=True)
    if safe_stderr:
        pieces.append(safe_stderr)
    if pieces:
        return " | ".join(pieces)
    safe_stdout = redact_sensitive_text((stdout or "").strip()[-1000:], force=True)
    return safe_stdout or f"exit code {returncode}"


class _ClaudeCodeChatCompletions:
    def __init__(self, client: "ClaudeCodeCLIClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _ClaudeCodeChatNamespace:
    def __init__(self, client: "ClaudeCodeCLIClient"):
        self.completions = _ClaudeCodeChatCompletions(client)


class _AsyncClaudeCodeChatCompletions:
    """Async shim around the synchronous Claude Code CLI subprocess facade."""

    def __init__(self, sync_completions: _ClaudeCodeChatCompletions):
        self._sync = sync_completions

    async def create(self, **kwargs: Any) -> Any:
        import asyncio

        return await asyncio.to_thread(self._sync.create, **kwargs)


class _AsyncClaudeCodeChatNamespace:
    def __init__(self, completions: _AsyncClaudeCodeChatCompletions):
        self.completions = completions


class AsyncClaudeCodeCLIClient:
    """Async-compatible wrapper matching ``AsyncOpenAI.chat.completions``."""

    def __init__(self, sync_client: "ClaudeCodeCLIClient"):
        self._sync_client = sync_client
        self.chat = _AsyncClaudeCodeChatNamespace(
            _AsyncClaudeCodeChatCompletions(sync_client.chat.completions)
        )
        self.api_key = sync_client.api_key
        self.base_url = sync_client.base_url
        # Match auxiliary wrappers that expose the underlying client for cache
        # eviction / poisoning logic.
        self._real_client = sync_client

    @property
    def is_closed(self) -> bool:
        return bool(getattr(self._sync_client, "is_closed", False))

    def close(self) -> None:
        self._sync_client.close()

    async def aclose(self) -> None:
        self._sync_client.close()


class ClaudeCodeCLIClient:
    """Minimal OpenAI-client-compatible facade around ``claude -p``."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        claude_command: str | None = None,
        claude_args: list[str] | None = None,
        cwd: str | None = None,
        timeout: Any = None,
        **_: Any,
    ):
        self.api_key = api_key or "claude-code-cli"
        self.base_url = base_url or CLAUDE_CODE_MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self._command = claude_command or command or _resolve_command()
        self._args = list(claude_args or args or _resolve_args())
        self._cwd = str(Path(cwd or os.getcwd()).resolve())
        self._timeout = timeout
        self.chat = _ClaudeCodeChatNamespace(self)
        self.is_closed = False
        self._active_process: subprocess.Popen[str] | None = None
        self._active_process_lock = threading.Lock()

    def close(self) -> None:
        proc: subprocess.Popen[str] | None
        with self._active_process_lock:
            proc = self._active_process
            self._active_process = None
        self.is_closed = True
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: Any = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        stream: bool = False,
        **_: Any,
    ) -> Any:
        prompt = _format_messages_as_prompt(
            messages or [], model=model, tools=tools, tool_choice=tool_choice
        )
        effective_timeout = self._coerce_timeout(timeout if timeout is not None else self._timeout)
        response_text, reasoning_text, usage, actual_model = self._run_prompt(
            prompt,
            model=model,
            timeout_seconds=effective_timeout,
        )
        tool_calls, cleaned_text = _extract_tool_calls_from_text(response_text)
        if stream:
            return self._stream_chunks(
                model=actual_model or model or "claude-code-cli",
                content=cleaned_text,
                reasoning=reasoning_text,
                tool_calls=tool_calls,
                usage=usage,
            )
        return self._final_response(
            model=actual_model or model or "claude-code-cli",
            content=cleaned_text,
            reasoning=reasoning_text,
            tool_calls=tool_calls,
            usage=usage,
        )

    @staticmethod
    def _coerce_timeout(timeout: Any) -> float:
        if timeout is None:
            return _DEFAULT_TIMEOUT_SECONDS
        if isinstance(timeout, (int, float)):
            return float(timeout)
        candidates = [
            getattr(timeout, attr, None)
            for attr in ("read", "write", "connect", "pool", "timeout")
        ]
        numeric = [float(v) for v in candidates if isinstance(v, (int, float))]
        return max(numeric) if numeric else _DEFAULT_TIMEOUT_SECONDS

    def _command_for_model(self, model: str | None) -> list[str]:
        args = list(self._args)
        if not _has_flag(args, "-p", "--print"):
            args.insert(0, "-p")
        if not _has_flag(args, "--output-format"):
            args.extend(["--output-format", "stream-json"])
        if not _has_flag(args, "--verbose"):
            args.append("--verbose")
        if not _has_flag(args, "--strict-mcp-config"):
            args.append("--strict-mcp-config")
        if not _has_flag(args, "--setting-sources"):
            args.extend(["--setting-sources", "user"])
        if not _has_flag(args, "--disallowedTools", "--disallowed-tools"):
            args.extend(["--disallowedTools", ",".join(_DEFAULT_DISALLOWED_TOOLS)])
        if not _has_flag(args, "--model") and not _is_model_placeholder(model):
            args.extend(["--model", str(model)])
        return [self._command] + args

    def _run_prompt(
        self,
        prompt_text: str,
        *,
        model: str | None,
        timeout_seconds: float,
    ) -> tuple[str, str, SimpleNamespace, str | None]:
        cmd = self._command_for_model(model)
        resolved = shutil.which(cmd[0]) if cmd and cmd[0] else None
        if not resolved:
            raise RuntimeError(
                "Could not find Claude Code CLI command "
                f"'{cmd[0] if cmd else self._command}'. Install it with "
                "`npm install -g @anthropic-ai/claude-code` or set "
                "HERMES_CLAUDE_CODE_COMMAND/CLAUDE_CODE_CLI_PATH."
            )
        cmd[0] = resolved
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self._cwd,
                env=_build_subprocess_env(),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"Could not start Claude Code CLI command '{cmd[0]}'.") from exc

        with self._active_process_lock:
            self._active_process = proc
        self.is_closed = False
        try:
            try:
                stdout, stderr = proc.communicate(prompt_text, timeout=timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                proc.kill()
                stdout, stderr = proc.communicate(timeout=5)
                raise TimeoutError(
                    f"Timed out waiting for Claude Code CLI after {timeout_seconds:.0f}s. "
                    f"stderr tail: {redact_sensitive_text((stderr or '')[-2000:], force=True)}"
                ) from exc
        finally:
            with self._active_process_lock:
                if self._active_process is proc:
                    self._active_process = None

        parsed = self._parse_output(stdout)
        if proc.returncode != 0:
            status = _extract_result_status(stdout)
            terminal_reason = str(status.get("terminal_reason") or "").strip()
            # Claude Code may exit non-zero when --max-turns was supplied and
            # the run hit that artificial ceiling, even though a complete
            # assistant message was already emitted.  Treat that specific shape
            # as a usable model response instead of poisoning Hermes' retry loop
            # with a raw stream-json usage blob.
            if terminal_reason == "max_turns" and str(parsed[0] or "").strip():
                return parsed
            detail = _summarize_failure(stderr or "", stdout or "", proc.returncode)
            raise RuntimeError(f"Claude Code CLI failed: {detail}")

        return parsed

    def _parse_output(self, stdout: str) -> tuple[str, str, SimpleNamespace, str | None]:
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        final_text = ""
        usage_raw: dict[str, Any] | None = None
        actual_model: str | None = None

        for raw_line in (stdout or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                text_parts.append(line)
                continue
            typ = obj.get("type")
            if typ == "system" and obj.get("subtype") == "init":
                if isinstance(obj.get("model"), str):
                    actual_model = obj.get("model")
                continue
            if typ == "stream_event":
                event = obj.get("event") or {}
                if not isinstance(event, dict):
                    continue
                if event.get("type") == "message_start":
                    msg = event.get("message") or {}
                    if isinstance(msg, dict):
                        actual_model = msg.get("model") or actual_model
                        if isinstance(msg.get("usage"), dict):
                            usage_raw = msg.get("usage")
                elif event.get("type") == "content_block_delta":
                    delta = event.get("delta") or {}
                    if isinstance(delta, dict):
                        if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                            text_parts.append(delta.get("text") or "")
                        elif "thinking" in delta and isinstance(delta.get("thinking"), str):
                            reasoning_parts.append(delta.get("thinking") or "")
                elif event.get("type") == "message_delta":
                    delta_usage = event.get("usage")
                    if isinstance(delta_usage, dict):
                        # Merge, don't overwrite. Anthropic message_delta usage
                        # typically carries only output_tokens, so a blind assign
                        # drops the input_tokens captured at message_start and
                        # leaves prompt_tokens=0 -- Hermes then thinks the context
                        # is empty, never compresses, and the CLI later rejects
                        # the prompt as too long. Prefer non-null delta fields.
                        if isinstance(usage_raw, dict):
                            for _k, _v in delta_usage.items():
                                if _v is not None:
                                    usage_raw[_k] = _v
                        else:
                            usage_raw = delta_usage
                continue
            if typ == "assistant":
                msg = obj.get("message") or {}
                if isinstance(msg, dict):
                    actual_model = msg.get("model") or actual_model
                    if isinstance(msg.get("usage"), dict):
                        usage_raw = msg.get("usage")
                    content = msg.get("content") or []
                    assistant_parts: list[str] = []
                    if isinstance(content, list):
                        for part in content:
                            if not isinstance(part, dict):
                                continue
                            if part.get("type") == "text" and isinstance(part.get("text"), str):
                                assistant_parts.append(part.get("text") or "")
                            elif part.get("type") in {"thinking", "redacted_thinking"}:
                                val = part.get("thinking") or part.get("text") or ""
                                if isinstance(val, str):
                                    reasoning_parts.append(val)
                    if assistant_parts:
                        final_text = "".join(assistant_parts)
                continue
            if typ == "result" and isinstance(obj.get("result"), str):
                final_text = obj.get("result") or final_text

        text = final_text if final_text else "".join(text_parts)
        return text, "".join(reasoning_parts), _namespace_usage(usage_raw), actual_model

    @staticmethod
    def _final_response(
        *,
        model: str,
        content: str,
        reasoning: str,
        tool_calls: list[SimpleNamespace],
        usage: SimpleNamespace,
    ) -> SimpleNamespace:
        message = SimpleNamespace(
            content=content or None,
            tool_calls=tool_calls or None,
            reasoning=reasoning or None,
            reasoning_content=reasoning or None,
            reasoning_details=None,
        )
        finish_reason = "tool_calls" if tool_calls else "stop"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
            usage=usage,
            model=model,
        )

    @staticmethod
    def _stream_chunks(
        *,
        model: str,
        content: str,
        reasoning: str,
        tool_calls: list[SimpleNamespace],
        usage: SimpleNamespace,
    ) -> Iterable[SimpleNamespace]:
        if reasoning:
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None,
                            reasoning_content=reasoning,
                            reasoning=reasoning,
                            tool_calls=None,
                        ),
                        finish_reason=None,
                    )
                ],
                model=model,
                usage=None,
            )
        if tool_calls:
            deltas = []
            for idx, call in enumerate(tool_calls):
                deltas.append(
                    SimpleNamespace(
                        index=idx,
                        id=call.id,
                        type="function",
                        function=SimpleNamespace(
                            name=call.function.name,
                            arguments=call.function.arguments,
                        ),
                    )
                )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=None, reasoning_content=None, tool_calls=deltas),
                        finish_reason="tool_calls",
                    )
                ],
                model=model,
                usage=None,
            )
        else:
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=content or "",
                            reasoning_content=None,
                            reasoning=None,
                            tool_calls=None,
                        ),
                        finish_reason="stop",
                    )
                ],
                model=model,
                usage=None,
            )
        yield SimpleNamespace(choices=[], model=model, usage=usage)
