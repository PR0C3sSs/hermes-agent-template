#!/usr/bin/env python3
"""Re-apply the Claude Code CLI provider overlay on Railway startup.

The Railway image stores Hermes under /opt/hermes-agent, while /data/.hermes is
persistent.  This script copies the small set of patched source files from the
persistent overlay into /opt before launching `hermes gateway`.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

try:
    import yaml
except Exception:  # pragma: no cover - Hermes ships PyYAML; fallback is explicit.
    yaml = None

PATCH_DIR = Path(__file__).resolve().parent
OVERLAY = PATCH_DIR / "overlay"
ROOT = Path(os.environ.get("HERMES_AGENT_ROOT", "/opt/hermes-agent"))
HERMES_HOME = Path(os.environ.get("HERMES_HOME", "/data/.hermes"))

FILES = [
    "agent/claude_code_cli_client.py",
    "agent/auxiliary_client.py",
    "agent/agent_runtime_helpers.py",
    "agent/agent_init.py",
    "agent/model_metadata.py",
    "hermes_cli/auth.py",
    "hermes_cli/runtime_provider.py",
    "hermes_cli/providers.py",
    "hermes_cli/models.py",
    "hermes_cli/model_switch.py",
    "hermes_cli/model_normalize.py",
    "plugins/model-providers/claude-code-cli/__init__.py",
    "plugins/model-providers/claude-code-cli/plugin.yaml",
]


def _same_bytes(a: Path, b: Path) -> bool:
    try:
        return a.read_bytes() == b.read_bytes()
    except FileNotFoundError:
        return False


def apply_overlay() -> list[str]:
    changed: list[str] = []
    if not ROOT.exists():
        raise RuntimeError(f"Hermes source root not found: {ROOT}")
    for rel in FILES:
        src = OVERLAY / rel
        dst = ROOT / rel
        if not src.exists():
            raise RuntimeError(f"Missing persisted overlay file: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if _same_bytes(src, dst):
            continue
        shutil.copy2(src, dst)
        changed.append(rel)
    return changed


def update_config() -> None:
    if os.environ.get("HERMES_CLAUDE_CODE_CLI_ENABLE", "1").strip().lower() in {"0", "false", "no", "off"}:
        return
    if yaml is None:
        raise RuntimeError("PyYAML is required to persist Claude Code CLI config")

    cfg_path = HERMES_HOME / "config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg: dict = {}
    if cfg_path.exists():
        try:
            loaded = yaml.safe_load(cfg_path.read_text())
            if isinstance(loaded, dict):
                cfg = loaded
        except Exception:
            cfg = {}

    model_name = (
        os.environ.get("CLAUDE_CODE_CLI_MODEL", "").strip()
        or os.environ.get("LLM_MODEL", "").strip()
        or "claude-opus-4-8"
    )
    provider = os.environ.get("LLM_PROVIDER", "claude-code-cli").strip() or "claude-code-cli"

    model_cfg = dict(cfg.get("model") if isinstance(cfg.get("model"), dict) else {})
    if provider == "claude-code-cli":
        model_cfg["provider"] = "claude-code-cli"
        model_cfg["default"] = model_name
        model_cfg["base_url"] = "claude-code-cli://claude"
        model_cfg["api_mode"] = "chat_completions"
        # Do NOT pin a context_length by default.  Treat claude-code-cli like
        # any other provider and let Hermes resolve the model's real advertised
        # window (see get_model_context_length).  Only pin when the operator
        # explicitly caps it via env, e.g. if a specific account/session
        # actually rejects long prompts and needs a manual ceiling.
        ctx_override = os.environ.get("CLAUDE_CODE_CLI_CONTEXT_LENGTH", "").strip()
        if ctx_override.isdigit() and int(ctx_override) > 0:
            model_cfg["context_length"] = int(ctx_override)
        else:
            model_cfg.pop("context_length", None)
    cfg["model"] = model_cfg

    # Do NOT pin auxiliary.compression to a second model/provider here.
    # Hermes' intended auto behavior is: use the active main provider/model
    # first, and only fall back elsewhere if that cannot be constructed.  The
    # overlay teaches auxiliary_client how to construct claude-code-cli, so
    # compression can now run through the same Claude Code CLI path instead of
    # introducing DeepSeek/OpenRouter/etc.  If a previous version of this patch
    # wrote a DeepSeek compression override, remove just that override.
    aux = cfg.get("auxiliary")
    if isinstance(aux, dict):
        comp = aux.get("compression")
        if isinstance(comp, dict) and str(comp.get("provider") or "").strip().lower() == "deepseek":
            aux.pop("compression", None)
        if not aux:
            cfg.pop("auxiliary", None)
    cfg.setdefault("data_dir", str(HERMES_HOME))

    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
    try:
        cfg_path.chmod(0o600)
    except Exception:
        pass


def main() -> int:
    changed = apply_overlay()
    update_config()
    if changed:
        print("[claude-code-cli patch] applied: " + ", ".join(changed), file=sys.stderr)
    else:
        print("[claude-code-cli patch] already applied", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
