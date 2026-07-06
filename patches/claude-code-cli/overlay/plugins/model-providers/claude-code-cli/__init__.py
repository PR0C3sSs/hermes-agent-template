"""Claude Code CLI provider profile.

This provider is an external subprocess transport: Hermes talks OpenAI-style
chat completions internally, while the client facade spawns `claude -p` with
stream-json output and the user's existing Claude Code login.
"""

from providers import register_provider
from providers.base import ProviderProfile


class ClaudeCodeCLIProfile(ProviderProfile):
    """Claude Code CLI — local subprocess, no REST models endpoint."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        return None


claude_code_cli = ClaudeCodeCLIProfile(
    name="claude-code-cli",
    aliases=("claude-cli", "claude-code-max"),
    api_mode="chat_completions",
    env_vars=(),
    base_url="claude-code-cli://claude",
    auth_type="external_process",
    display_name="Claude Code CLI",
    description="Claude Code CLI (spawns local `claude -p` and reuses Claude Max login)",
    fallback_models=(
        "claude-fable-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-opus-4-5",
        "claude-opus-4-1",
        "claude-opus-4-0",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5",
        "claude-sonnet-4-0",
        "claude-haiku-4-5",
        "claude-3-7-sonnet",
        "claude-3-5-sonnet",
        "claude-3-5-haiku",
        "claude-3-opus",
        "claude-3-sonnet",
        "claude-3-haiku",
        "claude-code-cli",
    ),
)

register_provider(claude_code_cli)
