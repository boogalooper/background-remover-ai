from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime"
HF_HOME = RUNTIME / "huggingface"
LOG_PATH = ROOT / "background_remover_ai.log"
CRASH_LOG_PATH = ROOT / "background_remover_ai_crash.log"


def configure_runtime_environment() -> None:
    """Keep Hugging Face downloads inside the application folder."""
    HF_HOME.mkdir(parents=True, exist_ok=True)
    # Deliberately override an inherited user/system HF_HOME.  Models and the
    # token for this application must move together with the project folder.
    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
