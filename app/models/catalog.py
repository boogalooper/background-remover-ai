from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    repo_id: str
    input_size: int
    gated: bool
    license_note: str
    description: str
    safe_cuda_batch: int
    approximate_download_mb: int


MODEL_SPECS: dict[str, ModelSpec] = {
    "bria_rmbg_2": ModelSpec(
        key="bria_rmbg_2",
        label="BRIA RMBG-2.0 — универсальная",
        repo_id="briaai/RMBG-2.0",
        input_size=1024,
        gated=True,
        license_note="CC BY-NC 4.0 — личное/некоммерческое использование",
        description="Универсальная модель. Хороший первый вариант для портретов и предметов.",
        safe_cuda_batch=2,
        approximate_download_mb=1000,
    ),
    "birefnet_hr_matting": ModelSpec(
        key="birefnet_hr_matting",
        label="BiRefNet HR Matting — максимум деталей",
        repo_id="ZhengPeng7/BiRefNet_HR-matting",
        input_size=2048,
        gated=False,
        license_note="MIT",
        description="Модель 2048 px для более аккуратного края, волос и сложного контура.",
        safe_cuda_batch=1,
        approximate_download_mb=450,
    ),
    "birefnet_portrait": ModelSpec(
        key="birefnet_portrait",
        label="BiRefNet Portrait — люди",
        repo_id="ZhengPeng7/BiRefNet-portrait",
        input_size=1024,
        gated=False,
        license_note="См. карточку модели BiRefNet",
        description="Специализированный вариант для людей и портретов.",
        safe_cuda_batch=2,
        approximate_download_mb=900,
    ),
}

LABEL_TO_KEY = {spec.label: key for key, spec in MODEL_SPECS.items()}


def get_model_spec(key: str) -> ModelSpec:
    try:
        return MODEL_SPECS[key]
    except KeyError as exc:
        raise ValueError(f"Unknown model: {key}") from exc


def resolve_batch_size(requested: int, spec: ModelSpec, device: str, safe_memory: bool) -> int:
    value = max(1, min(8, int(requested)))
    if device != "cuda":
        return 1
    if safe_memory:
        return min(value, spec.safe_cuda_batch)
    return value
