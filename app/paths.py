from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime"
HF_HOME = RUNTIME / "huggingface"
LOG_PATH = ROOT / "background_remover_ai.log"
CRASH_LOG_PATH = ROOT / "background_remover_ai_crash.log"


def configure_runtime_environment() -> None:
    """Keep Hugging Face downloads inside the application folder.

    An inherited HF_TOKEN is deliberately left untouched.  This allows a
    machine-wide/user HF_TOKEN to authorize gated models without storing a
    second token inside the application directory.
    """
    HF_HOME.mkdir(parents=True, exist_ok=True)
    # Deliberately override an inherited user/system HF_HOME so model downloads
    # remain portable with the application. Authentication may still come from
    # an inherited HF_TOKEN.
    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def get_hf_token() -> str | None:
    """Return an inherited Hugging Face token without ever logging it."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    # Older setups sometimes use this name. Supporting it costs nothing and
    # keeps the application's behavior friendly to existing machines.
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN", "").strip()
    return token or None
