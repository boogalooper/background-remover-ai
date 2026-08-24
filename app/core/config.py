from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from app.paths import ROOT

DEFAULT_CONFIG_PATH = ROOT / "config" / "default.json"
UI_STATE_PATH = ROOT / "config" / "ui_state.json"
CUSTOM_MASK_PRESETS_PATH = ROOT / "config" / "custom_mask_presets.json"


def load_config(path: Path | None = None) -> dict[str, Any]:
    source = path or DEFAULT_CONFIG_PATH
    with source.open("r", encoding="utf-8") as fh:
        value = json.load(fh)
    if not isinstance(value, dict):
        raise ValueError("Config root must be a JSON object")
    return value


def load_ui_state() -> dict[str, Any]:
    if not UI_STATE_PATH.exists():
        return {}
    try:
        with UI_STATE_PATH.open("r", encoding="utf-8") as fh:
            value = json.load(fh)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def save_ui_state(state: dict[str, Any]) -> None:
    UI_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = UI_STATE_PATH.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False, indent=2)
    tmp.replace(UI_STATE_PATH)




def load_custom_mask_presets() -> dict[str, dict[str, Any]]:
    if not CUSTOM_MASK_PRESETS_PATH.exists():
        return {}
    try:
        with CUSTOM_MASK_PRESETS_PATH.open("r", encoding="utf-8") as fh:
            value = json.load(fh)
        if not isinstance(value, dict):
            return {}
        result: dict[str, dict[str, Any]] = {}
        for name, settings in value.items():
            if isinstance(name, str) and name.strip() and isinstance(settings, dict):
                result[name.strip()] = settings
        return result
    except Exception:
        return {}


def save_custom_mask_presets(presets: dict[str, dict[str, Any]]) -> None:
    CUSTOM_MASK_PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CUSTOM_MASK_PRESETS_PATH.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(presets, fh, ensure_ascii=False, indent=2, sort_keys=True)
    tmp.replace(CUSTOM_MASK_PRESETS_PATH)


def merged_config(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    _deep_update(result, overrides)
    return result


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
