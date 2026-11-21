"""Run rewrite requests through a subscription CLI instead of a paid API.

`claude-cli` drives Claude Code in print mode (`claude -p`), signed in with a
Claude subscription token (CLAUDE_CODE_OAUTH_TOKEN from `claude setup-token`).
`codex-cli` drives `codex exec`, signed in with a ChatGPT plan (auth.json from
`codex login`). Neither bills per request; both count toward the plan's usage
limits. No API key is read here, and a set ANTHROPIC_API_KEY or
OPENAI_API_KEY stops the run, because either CLI would then bill the key.

Each client has the `chat(model, max_tokens, messages)` shape rewrite_bench
uses with OpenRouter. A conversation is one CLI session: the first call
starts it with the system prompt, and each later call in the same document
resumes it with only the newest user message, so the model sees the same
turns an API request would carry.

What differs from the API run, and is recorded on every row:
- The control arm, which sends no system prompt to the API, gets the one
  line NEUTRAL_SYSTEM here, since each CLI would otherwise use its own.
- Claude Code replaces its own system prompt with ours (--system-prompt-file),
  runs with no tools, no settings files, and no MCP servers, in an empty
  folder. max_tokens is not settable; the CLI's default applies.
- Codex replaces its base instructions with ours (model_instructions_file),
  but stays an agent: it still has its shell tool, here in a read-only
  sandbox in an empty folder. Its served model may differ from the API's.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

PAID_KEYS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")
TIMEOUT_S = 900

# Map the OpenRouter ids the bench already uses to each CLI's model names, so
# a subscription run can resume the same rewrite file.
CLAUDE_MODELS = {"anthropic/claude-sonnet-5.5": "claude-sonnet-5-5"}
CODEX_MODELS = {"openai/gpt-5.4": "gpt-5.4"}

NEUTRAL_SYSTEM = "You are a helpful assistant."
LIMIT_WORDS = ("usage limit", "rate limit", "limit reached", "too many requests", "429", "quota")


class UsageLimitError(Exception):
    """The plan's usage limit was hit. The run stops; finished docs are kept."""


class CLIError(Exception):
    """The CLI failed for another reason (sign-in, bad flag, crash)."""


def check_no_paid_keys(env: dict | None = None) -> None:
    env = os.environ if env is None else env
    found = [key for key in PAID_KEYS if env.get(key)]
    if found:
        raise CLIError(f"{', '.join(found)} is set; unset it so the CLI cannot bill an API key")


def _limit_hit(text: str) -> bool:
    low = text.lower()
    return any(word in low for word in LIMIT_WORDS)


def _response(text: str, model: str | None, stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)], model=model)


class _SessionClient:
    backend = ""

    def __init__(self, run=subprocess.run, workdir: Path | None = None):
        check_no_paid_keys()
        self._run = run
        self._dir = Path(workdir or tempfile.mkdtemp(prefix=f"bench-{self.backend}-"))
        self._session: str | None = None
        self.last_events: list = []

    def _split(self, messages: list[dict]) -> tuple[str | None, str, bool]:
        """(system, newest user message, resume?) for a message list."""
        # A request with no system prompt (the control arm) gets a one-line
        # neutral one, since each CLI would otherwise use its own agent prompt.
        system = next((m["content"] for m in messages if m["role"] == "system"), NEUTRAL_SYSTEM)
        resume = any(m["role"] == "assistant" for m in messages)
        if resume and not self._session:
            raise CLIError("a follow-up turn arrived with no session to resume")
        if not resume:
            self._session = None
        return system, messages[-1]["content"], resume

    def _exec(self, cmd: list[str], prompt: str, env: dict | None = None) -> str:
        proc = self._run(
            cmd, input=prompt, capture_output=True, text=True, timeout=TIMEOUT_S, cwd=self._dir,
            env={**os.environ, **(env or {})},
        )
        if proc.returncode != 0:
            detail = (proc.stderr or "")[-1500:] + (proc.stdout or "")[-1500:]
            if _limit_hit(detail):
                raise UsageLimitError(detail.strip()[-500:])
            raise CLIError(f"{cmd[0]} exited {proc.returncode}: {detail.strip()[-800:]}")
        return proc.stdout


class ClaudeCLIClient(_SessionClient):
    backend = "claude-cli"

    def __init__(self, binary: str = "claude", **kwargs):
        super().__init__(**kwargs)
        self._bin = binary

    def chat(self, model: str, max_tokens: int, messages: list[dict]):
        system, prompt, resume = self._split(messages)
        cmd = [
            self._bin, "-p", "--output-format", "json", "--model", CLAUDE_MODELS.get(model, model),
            "--tools", "", "--setting-sources", "", "--strict-mcp-config",
        ]
        if resume:
            cmd += ["--resume", self._session]
        sys_file = self._dir / "system.md"
        sys_file.write_text(system, encoding="utf-8")
        cmd += ["--system-prompt-file", str(sys_file)]
        out = self._exec(cmd, prompt)
        try:
            data = json.loads(out)
        except json.JSONDecodeError as error:
            raise CLIError(f"claude printed no JSON: {out[-300:]!r}") from error
        self.last_events = [data]
        text = data.get("result") or ""
        if data.get("is_error"):
            if _limit_hit(text):
                raise UsageLimitError(text[-500:])
            raise CLIError(f"claude reported an error: {text[-500:]}")
        self._session = data.get("session_id") or self._session
        usage = data.get("usage") or {}
        served = next(iter(data.get("modelUsage") or {}), None) or CLAUDE_MODELS.get(model, model)
        stop = "max_tokens" if data.get("stop_reason") == "max_tokens" else "end_turn"
        return _response(text, served, stop), {
            "prompt_tokens": (usage.get("input_tokens") or 0)
            + (usage.get("cache_read_input_tokens") or 0)
            + (usage.get("cache_creation_input_tokens") or 0),
            "cached_tokens": usage.get("cache_read_input_tokens"),
            "completion_tokens": usage.get("output_tokens"),
            "api_equivalent_usd": data.get("total_cost_usd"),
        }


class CodexCLIClient(_SessionClient):
    backend = "codex-cli"

    def __init__(self, binary: str = "codex", **kwargs):
        super().__init__(**kwargs)
        self._bin = binary

    def chat(self, model: str, max_tokens: int, messages: list[dict]):
        system, prompt, resume = self._split(messages)
        cmd = [self._bin, "exec"]
        if resume:
            cmd += ["resume", self._session]
        cmd += [
            "--json", "--skip-git-repo-check", "--ignore-rules", "-m", CODEX_MODELS.get(model, model),
            "-c", "approval_policy=\"never\"", "-c", "sandbox_mode=\"read-only\"",
        ]
        sys_file = self._dir / "system.md"
        sys_file.write_text(system, encoding="utf-8")
        cmd += ["-c", f"model_instructions_file=\"{sys_file}\""]
        cmd.append("-")
        out = self._exec(cmd, prompt)
        events = []
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        self.last_events = events
        texts, usage, limits = [], {}, None
        for event in events:
            kind = event.get("type", "")
            if kind == "thread.started" and event.get("thread_id"):
                self._session = event["thread_id"]
            item = event.get("item") or {}
            if kind == "item.completed" and item.get("type") == "agent_message":
                texts.append(item.get("text") or "")
            if kind == "turn.completed":
                usage = event.get("usage") or usage
            if kind in ("turn.failed", "error"):
                detail = json.dumps(event)[-500:]
                if _limit_hit(detail):
                    raise UsageLimitError(detail)
                raise CLIError(f"codex reported: {detail}")
            if "rate_limits" in event:
                limits = event["rate_limits"]
        if not texts:
            raise CLIError(f"codex returned no reply; last events: {json.dumps(events[-3:])[-600:]}")
        return _response(texts[-1].strip(), CODEX_MODELS.get(model, model)), {
            "prompt_tokens": usage.get("input_tokens"),
            "cached_tokens": usage.get("cached_input_tokens"),
            "completion_tokens": usage.get("output_tokens"),
            "rate_limits": limits,
        }


CLIENTS = {"claude-cli": ClaudeCLIClient, "codex-cli": CodexCLIClient}


def installed(backend: str) -> bool:
    return shutil.which({"claude-cli": "claude", "codex-cli": "codex"}[backend]) is not None
