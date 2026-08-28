from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
    best_for: str = ""
    caveats: str = ""
    speed_hint: str = ""
    detail_hint: str = ""
    compact_hint: str = ""
    preserves_aspect_ratio: bool = False
    revision: str | None = None
    recommended_mask: dict[str, Any] = field(default_factory=dict)
    recommended_cutout: dict[str, Any] = field(default_factory=dict)


MODEL_SPECS: dict[str, ModelSpec] = {
    "bria_rmbg_2": ModelSpec(
        key="bria_rmbg_2",
        label="BRIA RMBG-2.0 — универсальная",
        repo_id="briaai/RMBG-2.0",
        input_size=1024,
        gated=True,
        license_note="CC BY-NC 4.0 — личное/некоммерческое использование",
        description="Универсальная модель. Хороший первый вариант для портретов, товаров и большинства обычных фотографий.",
        safe_cuda_batch=2,
        approximate_download_mb=1000,
        best_for="Повседневные вырезки, предметы, портреты, быстрый первый прогон папки.",
        caveats="Требует один раз принять условия BRIA на Hugging Face. На самых сложных волосах и мехе уступает matting-вариантам BiRefNet.",
        speed_hint="Скорость: средняя.",
        detail_hint="Край: хороший универсальный, но не максимальный по деталям.",
        compact_hint="Универсальная для людей и предметов: хороший баланс качества и скорости. Для очень сложных волос и меха лучше BiRefNet Matting / HR Matting.",
        recommended_cutout={"decontaminate": True, "decontam_strength": 0.45},
    ),
    "birefnet": ModelSpec(
        key="birefnet",
        label="BiRefNet Standard — универсальная",
        repo_id="ZhengPeng7/BiRefNet",
        revision="e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4",
        input_size=1024,
        gated=False,
        license_note="См. карточку модели BiRefNet",
        description="Базовая универсальная BiRefNet 1024 px. Хороший баланс между качеством и скоростью.",
        safe_cuda_batch=2,
        approximate_download_mb=430,
        best_for="Обычные вырезки людей и предметов, когда нужна универсальная модель без сильного замедления.",
        caveats="На очень больших фото и сложных волосах имеет смысл попробовать HR / Matting варианты.",
        speed_hint="Скорость: средняя, чаще всего быстрее HR-моделей.",
        detail_hint="Край: чище базовых универсальных моделей, но не максимально детальный.",
        compact_hint="Универсальная BiRefNet для людей и предметов: баланс качества и скорости. Для больших кадров лучше HR, для волос и меха — Matting.",
        recommended_cutout={"decontaminate": True, "decontam_strength": 0.45},
    ),
    "birefnet_lite": ModelSpec(
        key="birefnet_lite",
        label="BiRefNet Lite — быстрый",
        repo_id="ZhengPeng7/BiRefNet_lite",
        revision="7838f1c3472f827cd8ce13ab5ccc2ce48077360f",
        input_size=1024,
        gated=False,
        license_note="См. карточку модели BiRefNet",
        description="Облегчённая версия для более быстрой пакетной обработки и умеренного расхода VRAM.",
        safe_cuda_batch=4,
        approximate_download_mb=220,
        best_for="Быстрые массовые вырезки, предварительный прогон больших папок, слабые GPU.",
        caveats="Край и мелкие детали обычно слабее, чем у Standard / HR / Matting.",
        speed_hint="Скорость: самая высокая среди моделей BiRefNet в этом скрипте.",
        detail_hint="Край: достаточный для большинства задач, но не лучший на волосах.",
        compact_hint="Самая быстрая и экономная BiRefNet для больших пакетов. Мелкие детали и волосы обычно слабее Standard / HR / Matting.",
        recommended_mask={"black_point": 0.03, "white_point": 0.97},
        recommended_cutout={"decontaminate": True, "decontam_strength": 0.5},
    ),
    "birefnet_portrait": ModelSpec(
        key="birefnet_portrait",
        label="BiRefNet Portrait — люди",
        repo_id="ZhengPeng7/BiRefNet-portrait",
        revision="ecdeb6240ef23557dbd48ff27c59c1a88cbcb755",
        input_size=1024,
        gated=False,
        license_note="См. карточку модели BiRefNet",
        description="Специализированный вариант для людей, лиц, волос и одежды на портретах.",
        safe_cuda_batch=2,
        approximate_download_mb=900,
        best_for="Портреты, полный рост, люди на нейтральном или относительно простом фоне.",
        caveats="Для предметов и сложных объектов иногда лучше Standard / HR; для экстремально сложных волос — Matting.",
        speed_hint="Скорость: средняя.",
        detail_hint="Край: хороший на людях, особенно на контурах одежды и головы.",
        compact_hint="Оптимизирована для людей: портрет, полный рост, волосы и одежда. Для предметов лучше Standard / HR; для сложных волос — Matting.",
        recommended_cutout={"decontaminate": True, "decontam_strength": 0.5},
    ),
    "birefnet_matting": ModelSpec(
        key="birefnet_matting",
        label="BiRefNet Matting — мягкий край",
        repo_id="ZhengPeng7/BiRefNet-matting",
        revision="57f9f68b43ba337c75762b14cf3075d659007268",
        input_size=1024,
        gated=False,
        license_note="См. карточку модели BiRefNet",
        description="Trimap-free matting-модель для волос, меха, фаты, прозрачных и полупрозрачных краёв.",
        safe_cuda_batch=1,
        approximate_download_mb=900,
        best_for="Волосы, мех, мягкие ткани, свадебные аксессуары, полупрозрачные края.",
        caveats="Обычно медленнее и тяжелее. Для простых предметов избыточна.",
        speed_hint="Скорость: ниже средней.",
        detail_hint="Край: заметно мягче и аккуратнее на сложных полупрозрачных границах.",
        compact_hint="Для волос, меха, фаты и полупрозрачных краёв: мягкий аккуратный matte. Медленнее и обычно избыточна для простых предметов.",
        recommended_cutout={"decontaminate": True, "decontam_strength": 0.6},
    ),
    "birefnet_hr": ModelSpec(
        key="birefnet_hr",
        label="BiRefNet HR — высокое разрешение",
        repo_id="ZhengPeng7/BiRefNet_HR",
        revision="a7a562f6fd16021180f2f4348f4de003a2d3d1e1",
        input_size=2048,
        gated=False,
        license_note="См. карточку модели BiRefNet",
        description="Универсальная HR-модель 2048 px. Лучше сохраняет мелкие детали на больших исходниках.",
        safe_cuda_batch=1,
        approximate_download_mb=450,
        best_for="Крупные фото, предметы со сложным контуром, детализированные объекты, когда 1024 уже мало.",
        caveats="Требует больше VRAM и заметно медленнее стандартных моделей.",
        speed_hint="Скорость: низкая.",
        detail_hint="Край: очень детальный на больших фото, но без особого упора на полупрозрачное matting-поведение.",
        compact_hint="2048 px для больших фото и мелких деталей сложного контура. Требовательнее к VRAM и заметно медленнее стандартных моделей.",
        recommended_cutout={"decontaminate": True, "decontam_strength": 0.5},
    ),
    "birefnet_hr_matting": ModelSpec(
        key="birefnet_hr_matting",
        label="BiRefNet HR Matting — максимум деталей",
        repo_id="ZhengPeng7/BiRefNet_HR-matting",
        revision="5d6b6f8adcb5b417c871b1d84ceaae9871355b7f",
        input_size=2048,
        gated=False,
        license_note="MIT",
        description="Matting-модель 2048 px для максимально аккуратного края, волос и сложного контура.",
        safe_cuda_batch=1,
        approximate_download_mb=450,
        best_for="Самые сложные волосы, мех, фата, размытые края, крупные исходники 20–60 Мп.",
        caveats="Самая тяжёлая и одна из самых медленных моделей. Для простых фото избыточна.",
        speed_hint="Скорость: низкая.",
        detail_hint="Край: лучший вариант в этом списке для сложных мягких границ.",
        compact_hint="2048 px для максимального качества волос, меха и мягких/размытых краёв. Самая тяжёлая и медленная; для простых сцен избыточна.",
        recommended_cutout={"decontaminate": True, "decontam_strength": 0.65},
    ),
    "birefnet_dynamic": ModelSpec(
        key="birefnet_dynamic",
        label="BiRefNet Dynamic — разные пропорции",
        repo_id="ZhengPeng7/BiRefNet_dynamic",
        revision="280306042f57b7a33854319da62fd86aaa89ec4c",
        input_size=2304,
        gated=False,
        license_note="См. карточку модели BiRefNet",
        description="Модель с динамическим входом: сохраняет пропорции кадра и лучше подходит для очень разных размеров и форматов.",
        safe_cuda_batch=1,
        approximate_download_mb=460,
        best_for="Смешанные папки с очень разными пропорциями, панорамы, вертикали, кадры без желания сжимать их в квадрат.",
        caveats="Медленнее стандартных моделей и тяжелее по VRAM. Из-за динамического размера пакет обычно приходится держать маленьким.",
        speed_hint="Скорость: ниже средней.",
        detail_hint="Край: хороший универсальный; главное преимущество — сохранение пропорций входа.",
        compact_hint="Сохраняет пропорции кадра и адаптирует размер входа; хороша для вертикалей, панорам и смешанных папок. Требовательнее к VRAM, обычно batch 1.",
        preserves_aspect_ratio=True,
        recommended_cutout={"decontaminate": True, "decontam_strength": 0.5},
    ),
}

LABEL_TO_KEY = {spec.label: key for key, spec in MODEL_SPECS.items()}
DEFAULT_MODEL_KEY = "bria_rmbg_2"


def get_model_spec(key: str) -> ModelSpec:
    try:
        return MODEL_SPECS[key]
    except KeyError as exc:
        raise ValueError(f"Unknown model: {key}") from exc


def resolve_batch_size(requested: int, spec: ModelSpec, device: str, safe_memory: bool) -> int:
    value = max(1, min(8, int(requested)))
    if device != "cuda":
        return 1
    # Dynamic BiRefNet preserves each image aspect ratio. Different shapes in
    # one tensor must otherwise be padded to the largest image in the batch,
    # changing its intended inference path and wasting VRAM. Keep it strictly
    # one image per model call regardless of the unsafe-memory toggle.
    if spec.preserves_aspect_ratio:
        return 1
    if safe_memory:
        return min(value, spec.safe_cuda_batch)
    return value


def model_recommended_overrides(key: str) -> dict[str, Any]:
    spec = get_model_spec(key)
    mask = {
        "black_point": 0.02,
        "white_point": 0.98,
        "gamma": 1.0,
        "expand_pixels": 0,
        "feather_radius": 0.0,
        "guided_refine": False,
        "guided_max_long_edge": 4096,
        "guided_radius": 8,
        "guided_blend": 0.35,
    }
    cutout = {
        "decontaminate": True,
        "decontam_strength": 0.5,
    }
    mask.update(spec.recommended_mask)
    cutout.update(spec.recommended_cutout)
    # These are strictly quality-related parameters.  Do not include file,
    # output, device, batching or other general settings here: the GUI button
    # must never reset unrelated user choices.
    return {
        "mask": mask,
        "cutout": cutout,
    }
