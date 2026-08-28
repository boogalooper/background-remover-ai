from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

from app.models.catalog import MODEL_SPECS
from app.paths import ROOT

DEFAULT_CONFIG_PATH = ROOT / "config" / "default.json"
UI_STATE_PATH = ROOT / "config" / "ui_state.json"
CUSTOM_MASK_PRESETS_PATH = ROOT / "config" / "custom_mask_presets.json"

_ALLOWED_OUTPUT_MODES = {"cutout", "mask", "both"}
_ALLOWED_CUTOUT_FORMATS = {"png", "tif", "tiff"}
_ALLOWED_DEVICES = {"auto", "cuda", "cpu"}
_INVALID_WINDOWS_FILENAME_CHARS = set('<>:"/\\|?*')


class ConfigValidationError(ValueError):
    """Raised when a configuration can lead to invalid or unsafe processing."""


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


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ConfigValidationError(f"Раздел настроек «{name}» отсутствует или имеет неверный формат.")
    return value


def _finite_number(section: dict[str, Any], key: str, *, minimum: float, maximum: float) -> float:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigValidationError(f"Параметр «{key}» должен быть числом.")
    value = float(value)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ConfigValidationError(f"Параметр «{key}» должен быть в диапазоне {minimum}…{maximum}.")
    return value


def _integer(section: dict[str, Any], key: str, *, minimum: int, maximum: int) -> int:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigValidationError(f"Параметр «{key}» должен быть целым числом.")
    if not minimum <= value <= maximum:
        raise ConfigValidationError(f"Параметр «{key}» должен быть в диапазоне {minimum}…{maximum}.")
    return value


def _validate_suffix(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ConfigValidationError(f"{label} не может быть пустым.")
    if any(ord(ch) < 32 or ch in _INVALID_WINDOWS_FILENAME_CHARS for ch in value):
        raise ConfigValidationError(
            f"{label} содержит символ, недопустимый в имени файла Windows: {value!r}."
        )
    if value.endswith((" ", ".")):
        raise ConfigValidationError(f"{label} не должен оканчиваться пробелом или точкой.")


def validate_config(config: dict[str, Any]) -> None:
    """Validate the complete runtime configuration without mutating it.

    The GUI uses the same validator as the batch pipeline, so a hand-edited JSON
    file cannot bypass the safety/range checks enforced by interface controls.
    Unknown extra keys are tolerated for forward/backward compatibility.
    """
    if not isinstance(config, dict):
        raise ConfigValidationError("Корень конфигурации должен быть JSON-объектом.")

    model = _section(config, "model")
    files = _section(config, "files")
    mask = _section(config, "mask")
    cutout = _section(config, "cutout")
    performance = _section(config, "performance")

    model_key = model.get("key")
    if not isinstance(model_key, str) or model_key not in MODEL_SPECS:
        raise ConfigValidationError(f"Неизвестная модель: {model_key!r}.")

    output_mode = files.get("output_mode")
    if output_mode not in _ALLOWED_OUTPUT_MODES:
        raise ConfigValidationError(f"Неизвестный режим результата: {output_mode!r}.")
    cutout_format = str(files.get("cutout_format", "")).lower()
    if cutout_format not in _ALLOWED_CUTOUT_FORMATS:
        raise ConfigValidationError(f"Неизвестный формат прозрачного файла: {cutout_format!r}.")
    _validate_suffix(files.get("cutout_suffix"), "Суффикс прозрачного файла")
    _validate_suffix(files.get("mask_suffix"), "Суффикс маски")

    black = _finite_number(mask, "black_point", minimum=0.0, maximum=1.0)
    white = _finite_number(mask, "white_point", minimum=0.0, maximum=1.0)
    if black >= white:
        raise ConfigValidationError("Для маски должно выполняться: 0 ≤ нижний порог < верхний порог ≤ 1.")
    _finite_number(mask, "gamma", minimum=0.3, maximum=3.0)
    _integer(mask, "expand_pixels", minimum=-100, maximum=100)
    _finite_number(mask, "feather_radius", minimum=0.0, maximum=10.0)
    _integer(mask, "guided_max_long_edge", minimum=1024, maximum=8192)
    _integer(mask, "guided_radius", minimum=1, maximum=32)
    _finite_number(mask, "guided_blend", minimum=0.0, maximum=1.0)
    _finite_number(cutout, "decontam_strength", minimum=0.0, maximum=1.0)

    device = performance.get("device")
    if device not in _ALLOWED_DEVICES:
        raise ConfigValidationError(f"Неизвестное устройство вычислений: {device!r}.")
    _integer(performance, "gpu_batch_size", minimum=1, maximum=8)
    _integer(performance, "prefetch_workers", minimum=1, maximum=8)
    _integer(performance, "prefetch_buffer", minimum=1, maximum=12)
