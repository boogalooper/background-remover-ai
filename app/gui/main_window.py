from __future__ import annotations

import queue
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

from app import __version__
from app.core.config import (
    load_custom_mask_presets,
    load_ui_state,
    merged_config,
    save_custom_mask_presets,
    save_ui_state,
)
from app.core.pipeline import BatchPipeline, CancelledError
from app.core.scanner import suggested_output_dir
from app.gui.scrollable import ScrollableFrame
from app.gui.tooltip import ToolTip
from app.models.catalog import DEFAULT_MODEL_KEY, LABEL_TO_KEY, MODEL_SPECS, get_model_spec, model_recommended_overrides
from app.models.downloader import download_all_models
from app.paths import HF_HOME, ROOT, get_hf_token

MODEL_LABELS = [spec.label for spec in MODEL_SPECS.values()]
OUTPUT_MODES = {
    "PNG/TIFF с прозрачностью": "cutout",
    "Только чёрно-белая маска": "mask",
    "Прозрачный файл + маска": "both",
}
OUTPUT_MODES_INV = {v: k for k, v in OUTPUT_MODES.items()}
FORMATS = {"PNG": "png", "TIFF с альфа-каналом": "tiff"}
FORMATS_INV = {v: k for k, v in FORMATS.items()}
PROFILE_RECOMMENDED = "Рекомендуемый для выбранной модели"
PROFILE_NATURAL = "Естественный край"
PROFILE_CLEAN = "Чище фон"
PROFILE_NONE = "Без дополнительной обработки"
PROFILE_CUSTOM = "Пользовательский"
USER_PROFILE_PREFIX = "Мой пресет: "

_BASE_QUALITY = {
    "mask": {
        "black_point": 0.02,
        "white_point": 0.98,
        "gamma": 1.0,
        "expand_pixels": 0,
        "feather_radius": 0.0,
        "guided_refine": False,
        "guided_max_long_edge": 4096,
        "guided_radius": 8,
        "guided_blend": 0.35,
    },
    "cutout": {
        "decontaminate": True,
        "decontam_strength": 0.5,
    },
}

EDGE_PROFILES = {
    PROFILE_RECOMMENDED: None,
    PROFILE_NATURAL: {
        "mask": dict(_BASE_QUALITY["mask"]),
        "cutout": dict(_BASE_QUALITY["cutout"]),
    },
    PROFILE_CLEAN: {
        "mask": {**_BASE_QUALITY["mask"], "black_point": 0.05, "white_point": 0.95},
        "cutout": dict(_BASE_QUALITY["cutout"]),
    },
    PROFILE_NONE: {
        "mask": {**_BASE_QUALITY["mask"], "black_point": 0.0, "white_point": 1.0},
        "cutout": {"decontaminate": False, "decontam_strength": 0.5},
    },
    PROFILE_CUSTOM: None,
}


class MainWindow(tk.Tk):
    def __init__(self, config: dict, initial_sources: list[str] | None = None, initial_output: str | None = None):
        super().__init__()
        self.title(f"Background Remover AI v{__version__}")
        screen_h = max(650, int(self.winfo_screenheight()))
        # Reserve enough vertical space for the longest model description on a
        # normal 1080p display, but still fit smaller screens. The progress area
        # itself is fixed outside the scrollable tabs below.
        h = max(620, min(900, screen_h - 90))
        self.geometry(f"1080x{h}")
        self.minsize(900, min(620, h))

        self.base_config = config
        self.state = load_ui_state()
        self.custom_edge_profiles = self._validated_custom_profiles(load_custom_mask_presets())
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.cancel_event = threading.Event()
        self._guided_widgets: list[tk.Widget] = []
        self._decontam_widgets: list[tk.Widget] = []
        self._edge_profile_combos: list[ttk.Combobox] = []
        self._applying_edge_profile = False

        saved = lambda k, d: self.state.get(k, d)
        default_model_key = saved("model_key", config["model"].get("key", DEFAULT_MODEL_KEY))
        default_spec = get_model_spec(default_model_key if default_model_key in MODEL_SPECS else DEFAULT_MODEL_KEY)

        self.sources_var = tk.StringVar(value="")
        self.output_var = tk.StringVar(value=saved("output", ""))
        self.model_label_var = tk.StringVar(value=default_spec.label)
        self.output_mode_label_var = tk.StringVar(value=OUTPUT_MODES_INV.get(saved("output_mode", config["files"].get("output_mode", "cutout")), "PNG/TIFF с прозрачностью"))
        self.format_label_var = tk.StringVar(value=FORMATS_INV.get(saved("cutout_format", config["files"].get("cutout_format", "png")), "PNG"))
        self.edge_profile_var = tk.StringVar(value=saved("edge_profile", PROFILE_RECOMMENDED))
        self.recursive_var = tk.BooleanVar(value=bool(saved("recursive", config["files"].get("recursive", True))))
        self.preserve_structure_var = tk.BooleanVar(value=bool(saved("preserve_structure", config["files"].get("preserve_structure", True))))
        self.overwrite_var = tk.BooleanVar(value=bool(saved("overwrite", config["files"].get("overwrite", False))))
        self.preserve_metadata_var = tk.BooleanVar(value=bool(saved("preserve_metadata", config["files"].get("preserve_metadata", True))))
        self.black_point_var = tk.DoubleVar(value=float(saved("black_point", config["mask"].get("black_point", 0.02))))
        self.white_point_var = tk.DoubleVar(value=float(saved("white_point", config["mask"].get("white_point", 0.98))))
        self.gamma_var = tk.DoubleVar(value=float(saved("gamma", config["mask"].get("gamma", 1.0))))
        self.expand_var = tk.IntVar(value=int(saved("expand_pixels", config["mask"].get("expand_pixels", 0))))
        self.feather_var = tk.DoubleVar(value=float(saved("feather_radius", config["mask"].get("feather_radius", 0.0))))
        self.guided_var = tk.BooleanVar(value=bool(saved("guided_refine", config["mask"].get("guided_refine", False))))
        self.guided_long_var = tk.IntVar(value=int(saved("guided_max_long_edge", config["mask"].get("guided_max_long_edge", 4096))))
        self.guided_radius_var = tk.IntVar(value=int(saved("guided_radius", config["mask"].get("guided_radius", 8))))
        self.guided_blend_var = tk.DoubleVar(value=float(saved("guided_blend", config["mask"].get("guided_blend", 0.35))))
        self.decontaminate_var = tk.BooleanVar(value=bool(saved("decontaminate", config.get("cutout", {}).get("decontaminate", True))))
        self.decontam_strength_var = tk.DoubleVar(value=float(saved("decontam_strength", config.get("cutout", {}).get("decontam_strength", 0.45))))
        self.device_var = tk.StringVar(value=saved("device", config["performance"].get("device", "auto")))
        self.fp16_var = tk.BooleanVar(value=bool(saved("fp16", config["performance"].get("fp16", True))))
        self.safe_memory_var = tk.BooleanVar(value=bool(saved("safe_gpu_memory", config["performance"].get("safe_gpu_memory", True))))
        self.gpu_batch_var = tk.IntVar(value=int(saved("gpu_batch_size", config["performance"].get("gpu_batch_size", 2))))
        self.prefetch_var = tk.IntVar(value=int(saved("prefetch_workers", config["performance"].get("prefetch_workers", 2))))
        self.prefetch_buffer_var = tk.IntVar(value=int(saved("prefetch_buffer", config["performance"].get("prefetch_buffer", 3))))
        self.cutout_suffix_var = tk.StringVar(value=saved("cutout_suffix", config["files"].get("cutout_suffix", "_cutout")))
        self.mask_suffix_var = tk.StringVar(value=saved("mask_suffix", config["files"].get("mask_suffix", "_mask")))

        self.source_paths: list[Path] = []
        self._output_autofilled_from_sources = False
        if initial_sources:
            self.source_paths = [Path(p).expanduser() for p in initial_sources]
            if initial_output:
                self.output_var.set(str(Path(initial_output).expanduser()))
            else:
                suggested = suggested_output_dir(self.source_paths)
                if suggested is not None:
                    self.output_var.set(str(suggested))
                    self._output_autofilled_from_sources = True

        self.active_model_var = tk.StringVar(value="Модель не запущена")
        self.status_var = tk.StringVar(value="Выберите папку или фотографии")
        self.detail_var = tk.StringVar(value="")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.percent_var = tk.StringVar(value="0.0%")

        self._build_ui()
        self.model_label_var.trace_add("write", lambda *_: self._on_model_changed())
        self.edge_profile_var.trace_add("write", lambda *_: self._update_delete_preset_state())
        self.output_mode_label_var.trace_add("write", lambda *_: self._apply_context_states())
        self.guided_var.trace_add("write", lambda *_: self._apply_context_states())
        self.decontaminate_var.trace_add("write", lambda *_: self._apply_context_states())
        for variable in (
            self.black_point_var,
            self.white_point_var,
            self.gamma_var,
            self.expand_var,
            self.feather_var,
            self.guided_var,
            self.guided_long_var,
            self.guided_radius_var,
            self.guided_blend_var,
            self.decontaminate_var,
            self.decontam_strength_var,
        ):
            variable.trace_add("write", lambda *_: self._on_quality_setting_changed())
        self._refresh_source_text()
        self._synchronize_edge_profile()
        self._on_model_changed()
        self._apply_context_states()
        self._update_delete_preset_state()
        self.after(100, self._poll_events)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        notebook = ttk.Notebook(root)
        notebook.grid(row=0, column=0, sticky="nsew")

        quick_scroll = ScrollableFrame(notebook)
        mask_scroll = ScrollableFrame(notebook)
        general_scroll = ScrollableFrame(notebook)
        notebook.add(quick_scroll, text="Быстрый старт")
        notebook.add(mask_scroll, text="Маска и край")
        notebook.add(general_scroll, text="Общие настройки")
        quick = quick_scroll.inner
        mask_tab = mask_scroll.inner
        general = general_scroll.inner
        quick.columnconfigure(1, weight=1)
        mask_tab.columnconfigure(1, weight=1)
        general.columnconfigure(1, weight=1)

        # Quick start: keep the normal workflow compact and familiar.
        r = 0
        title = ttk.Label(quick, text="Пакетное удаление фона", font=("Segoe UI", 16, "bold"))
        title.grid(row=r, column=0, columnspan=3, sticky="w", pady=(4, 12)); r += 1

        source_box = ttk.LabelFrame(quick, text="1. Исходные фотографии", padding=10)
        source_box.grid(row=r, column=0, columnspan=3, sticky="ew", pady=5); source_box.columnconfigure(0, weight=1); r += 1
        ttk.Entry(source_box, textvariable=self.sources_var, state="readonly").grid(row=0, column=0, columnspan=3, sticky="ew", padx=(0, 8))
        ttk.Button(source_box, text="Выбрать папку...", command=self._choose_folder).grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Button(source_box, text="Выбрать файлы...", command=self._choose_files).grid(row=1, column=1, sticky="w", pady=(8, 0), padx=(8, 0))
        ttk.Button(source_box, text="Очистить", command=self._clear_sources).grid(row=1, column=2, sticky="e", pady=(8, 0))
        ttk.Checkbutton(source_box, text="Включать подпапки", variable=self.recursive_var).grid(row=2, column=0, sticky="w", pady=(8, 0))

        out_box = ttk.LabelFrame(quick, text="2. Результат", padding=10)
        out_box.grid(row=r, column=0, columnspan=3, sticky="ew", pady=5); out_box.columnconfigure(0, weight=1); r += 1
        ttk.Entry(out_box, textvariable=self.output_var).grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(out_box, text="Выбрать...", command=self._choose_output).grid(row=0, column=1, sticky="e")
        ttk.Checkbutton(out_box, text="Сохранять структуру подпапок", variable=self.preserve_structure_var).grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Checkbutton(out_box, text="Перезаписывать готовые файлы", variable=self.overwrite_var).grid(row=1, column=1, sticky="w", pady=(8, 0))

        settings = ttk.LabelFrame(quick, text="3. Как обрабатывать", padding=10)
        settings.grid(row=r, column=0, columnspan=3, sticky="ew", pady=5); settings.columnconfigure(1, weight=1); r += 1
        ttk.Label(settings, text="Модель:").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=4)
        self.model_combo = ttk.Combobox(settings, textvariable=self.model_label_var, values=MODEL_LABELS, state="readonly", width=48)
        self.model_combo.grid(row=0, column=1, sticky="ew", pady=4)
        self.model_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_model_changed())

        # Keep model-description height stable. Without this, changing to a model
        # with a longer explanation changes the requested size of the whole tab
        # and can move the progress/status area out of view.
        hint_frame = ttk.Frame(settings, height=230)
        hint_frame.grid(row=1, column=1, sticky="ew", pady=(0, 6))
        hint_frame.grid_propagate(False)
        hint_frame.pack_propagate(False)
        self.model_hint = ttk.Label(hint_frame, text="", wraplength=700, justify="left", anchor="nw")
        self.model_hint.pack(fill="both", expand=True)

        ttk.Label(settings, text="Выход:").grid(row=2, column=0, sticky="w", padx=(0, 10), pady=4)
        out_combo = ttk.Combobox(settings, textvariable=self.output_mode_label_var, values=list(OUTPUT_MODES), state="readonly")
        out_combo.grid(row=2, column=1, sticky="ew", pady=4)
        out_combo.bind("<<ComboboxSelected>>", lambda _e: self._apply_context_states())
        ttk.Label(settings, text="Формат прозрачного файла:").grid(row=3, column=0, sticky="w", padx=(0, 10), pady=4)
        ttk.Combobox(settings, textvariable=self.format_label_var, values=list(FORMATS), state="readonly").grid(row=3, column=1, sticky="ew", pady=4)
        ttk.Label(settings, text="Профиль края:").grid(row=4, column=0, sticky="w", padx=(0, 10), pady=4)
        edge_quick = ttk.Combobox(settings, textvariable=self.edge_profile_var, values=self._edge_profile_values(), state="readonly")
        edge_quick.grid(row=4, column=1, sticky="ew", pady=4)
        edge_quick.bind("<<ComboboxSelected>>", lambda _e: self._apply_edge_profile())
        self._edge_profile_combos.append(edge_quick)
        ToolTip(
            edge_quick,
            "«Рекомендуемый для выбранной модели» применяет модель-зависимые параметры. Сохранённые пользовательские пресеты также появляются в этом списке; ручная правка несохранённых параметров переводит профиль в «Пользовательский».",
        )

        buttons = ttk.Frame(quick)
        buttons.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(14, 4)); r += 1
        buttons.columnconfigure(0, weight=1)
        self.start_button = ttk.Button(buttons, text="НАЧАТЬ ОБРАБОТКУ", command=self._start)
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 8), ipady=8)
        self.cancel_button = ttk.Button(buttons, text="Отмена", command=self._cancel, state="disabled")
        self.cancel_button.grid(row=0, column=1, sticky="e", ipady=8)

        # Mask tab: all quality controls live here. The recommended model profile
        # replaces the former standalone recommendation button.
        r = 0
        mask_box = ttk.LabelFrame(mask_tab, text="Маска и край", padding=10)
        mask_box.grid(row=r, column=0, columnspan=3, sticky="ew", pady=5); mask_box.columnconfigure(1, weight=1); r += 1
        ttk.Label(mask_box, text="Профиль края:").grid(row=0, column=0, sticky="w", padx=(0, 12), pady=3)
        edge_mask = ttk.Combobox(mask_box, textvariable=self.edge_profile_var, values=self._edge_profile_values(), state="readonly", width=42)
        edge_mask.grid(row=0, column=1, sticky="w", pady=3)
        edge_mask.bind("<<ComboboxSelected>>", lambda _e: self._apply_edge_profile())
        self._edge_profile_combos.append(edge_mask)
        ToolTip(
            edge_mask,
            "«Рекомендуемый для выбранной модели» применяет полный набор рекомендуемых параметров маски и края. "
            "Сохранённые пользовательские пресеты восстанавливают все параметры этой вкладки.",
        )
        preset_buttons = ttk.Frame(mask_box)
        preset_buttons.grid(row=0, column=2, sticky="w", padx=(12, 0), pady=3)
        self.save_preset_button = ttk.Button(preset_buttons, text="Создать пресет из текущих", command=self._create_custom_edge_profile)
        self.save_preset_button.grid(row=0, column=0, sticky="w")
        self.delete_preset_button = ttk.Button(preset_buttons, text="Удалить пресет", command=self._delete_custom_edge_profile)
        self.delete_preset_button.grid(row=0, column=1, sticky="w", padx=(8, 0))
        ToolTip(self.save_preset_button, "Сохраняет текущие параметры вкладки «Маска и край» как именованный пользовательский пресет.")
        ToolTip(self.delete_preset_button, "Удаляет только выбранный пользовательский пресет. Встроенные профили удалить нельзя.")
        self._spinrow(mask_box, 1, "Удалять слабый фон ниже:", self.black_point_var, 0.0, 0.5, 0.01, "Чем выше, тем сильнее очищаются полупрозрачные остатки фона.")
        self._spinrow(mask_box, 2, "Считать объектом выше:", self.white_point_var, 0.5, 1.0, 0.01, "Чем ниже, тем быстрее полупрозрачные участки становятся непрозрачными.")
        self._spinrow(mask_box, 3, "Гамма маски:", self.gamma_var, 0.3, 3.0, 0.05, "1.0 — без изменения. Используйте только для тонкой настройки края.")
        self._spinrow(mask_box, 4, "Сдвиг края маски, px (итоговый файл):", self.expand_var, -100, 100, 1, "Положительное значение расширяет объект, отрицательное — сжимает. Значение измеряется в пикселях итогового полноразмерного изображения; на 30–60 Мп фото 2–5 px почти незаметны, для проверки попробуйте ±20…40 px.")
        self._spinrow(mask_box, 5, "Размытие края, px:", self.feather_var, 0.0, 10.0, 0.25, "Обычно лучше оставить 0: современные модели уже дают мягкий край.")
        gcheck = ttk.Checkbutton(mask_box, text="Экспериментально уточнять край по исходному изображению", variable=self.guided_var, command=self._apply_context_states)
        gcheck.grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 3))
        self._guided_widgets += self._spinrow(mask_box, 7, "Макс. размер уточнения, px:", self.guided_long_var, 1024, 8192, 256, "Ограничивает расход RAM при работе с 40–60 Мп файлами.")
        self._guided_widgets += self._spinrow(mask_box, 8, "Радиус уточнения:", self.guided_radius_var, 1, 32, 1, "Больший радиус сильнее привязывает маску к крупным границам изображения.")
        self._guided_widgets += self._spinrow(mask_box, 9, "Сила уточнения:", self.guided_blend_var, 0.0, 1.0, 0.05, "0 — выключено, 1 — максимальное влияние. Рекомендуется около 0.35.")
        self.decontam_check = ttk.Checkbutton(mask_box, text="Очищать цвет полупрозрачного края в прозрачном файле", variable=self.decontaminate_var, command=self._apply_context_states)
        self.decontam_check.grid(row=10, column=0, columnspan=2, sticky="w", pady=(10, 3))
        ToolTip(self.decontam_check, "Уменьшает цветную кайму на волосах и полупрозрачных краях. Нужна только при записи прозрачного файла, на маску не влияет.")
        self._decontam_widgets += self._spinrow(mask_box, 11, "Сила очистки цвета края:", self.decontam_strength_var, 0.0, 1.0, 0.05, "0 — выключено, 1 — максимально агрессивная очистка каймы. Обычно достаточно 0.45–0.65.")

        # General settings: model downloads, files, performance and application-wide behaviour.
        r = 0
        model_box = ttk.LabelFrame(general, text="Управление моделями", padding=10)
        model_box.grid(row=r, column=0, columnspan=3, sticky="ew", pady=5); model_box.columnconfigure(1, weight=1); r += 1
        ttk.Label(
            model_box,
            text="Рабочая модель выбирается во вкладке «Быстрый старт». Здесь можно только заранее подготовить локальный кэш моделей.",
            wraplength=830,
            justify="left",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))

        total_model_mb = sum(spec.approximate_download_mb for spec in MODEL_SPECS.values())
        ttk.Label(
            model_box,
            text=(
                f"Можно заранее загрузить все {len(MODEL_SPECS)} моделей. Максимальный объём — примерно "
                f"{total_model_mb / 1024:.1f} ГБ; уже имеющиеся в кэше файлы повторно не скачиваются. "
                f"Модели сохраняются в {HF_HOME}."
            ),
            wraplength=830,
            justify="left",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self.download_models_button = ttk.Button(model_box, text="Скачать все модели", command=self._download_all_models)
        self.download_models_button.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ToolTip(
            self.download_models_button,
            "Скачиваются только файлы, нужные этому скрипту. GPU и веса моделей в RAM не загружаются. Для BRIA автоматически используется HF_TOKEN, если он задан.",
        )

        bria = ttk.LabelFrame(general, text="BRIA RMBG-2.0", padding=10)
        bria.grid(row=r, column=0, columnspan=3, sticky="ew", pady=5); r += 1
        self.bria_info_label = ttk.Label(bria, text="BRIA требует один раз принять условия некоммерческого использования на Hugging Face.", wraplength=830, justify="left")
        self.bria_info_label.grid(row=0, column=0, sticky="w")
        self.bria_setup_button = ttk.Button(bria, text="Настроить доступ к BRIA...", command=self._setup_bria)
        self.bria_setup_button.grid(row=1, column=0, sticky="w", pady=(8, 0))

        perf = ttk.LabelFrame(general, text="Производительность и память", padding=10)
        perf.grid(row=r, column=0, columnspan=3, sticky="ew", pady=5); perf.columnconfigure(1, weight=1); r += 1
        ttk.Label(perf, text="Устройство:").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Combobox(perf, textvariable=self.device_var, values=("auto", "cuda", "cpu"), state="readonly", width=18).grid(row=0, column=1, sticky="w", pady=4)
        ttk.Checkbutton(perf, text="FP16 на NVIDIA (быстрее и экономнее VRAM)", variable=self.fp16_var).grid(row=1, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Checkbutton(perf, text="Безопасно ограничивать GPU-пакет для выбранной модели", variable=self.safe_memory_var).grid(row=2, column=0, columnspan=2, sticky="w", pady=4)
        self._spinrow(perf, 3, "Кадров за один GPU-проход:", self.gpu_batch_var, 1, 8, 1, "Это один экземпляр модели с батчем, а не несколько копий модели в VRAM.")
        self._spinrow(perf, 4, "Потоки чтения файлов:", self.prefetch_var, 1, 8, 1, "Пока GPU считает текущий пакет, CPU заранее читает следующие изображения.")
        self._spinrow(perf, 5, "Буфер предзагрузки:", self.prefetch_buffer_var, 1, 12, 1, "Большой буфер расходует больше RAM. Для очень больших фото 3–4 обычно достаточно.")

        files_box = ttk.LabelFrame(general, text="Файлы и сохранение", padding=10)
        files_box.grid(row=r, column=0, columnspan=3, sticky="ew", pady=5); files_box.columnconfigure(1, weight=1); r += 1
        ttk.Checkbutton(files_box, text="Сохранять доступные ICC/EXIF/DPI в результате", variable=self.preserve_metadata_var).grid(row=0, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Label(files_box, text="Суффикс прозрачного файла:").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(files_box, textvariable=self.cutout_suffix_var, width=20).grid(row=1, column=1, sticky="w", pady=4)
        ttk.Label(files_box, text="Суффикс маски:").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(files_box, textvariable=self.mask_suffix_var, width=20).grid(row=2, column=1, sticky="w", pady=4)

        # Persistent footer: progress and the actually used model remain visible
        # even when a tab needs scrolling or a model has a long description.
        progress_box = ttk.LabelFrame(root, text="Ход работы", padding=10)
        progress_box.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        progress_box.columnconfigure(0, weight=1)
        ttk.Label(progress_box, textvariable=self.active_model_var, font=("Segoe UI", 9, "bold"), wraplength=940).grid(
            row=0, column=0, sticky="w", pady=(0, 4)
        )
        line = ttk.Frame(progress_box)
        line.grid(row=1, column=0, sticky="ew")
        line.columnconfigure(0, weight=1)
        ttk.Label(line, textvariable=self.status_var, wraplength=860).grid(row=0, column=0, sticky="w")
        ttk.Label(line, textvariable=self.percent_var).grid(row=0, column=1, sticky="e")
        ttk.Progressbar(progress_box, variable=self.progress_var, maximum=100).grid(row=2, column=0, sticky="ew", pady=(6, 4))
        ttk.Label(progress_box, textvariable=self.detail_var, wraplength=940).grid(row=3, column=0, sticky="w")

    def _spinrow(self, parent, row, label, variable, lo, hi, increment, tip) -> list[tk.Widget]:
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=row, column=0, sticky="w", padx=(0, 12), pady=3)
        spin = ttk.Spinbox(parent, textvariable=variable, from_=lo, to=hi, increment=increment, width=12)
        spin.grid(row=row, column=1, sticky="w", pady=3)
        ToolTip(lbl, tip); ToolTip(spin, tip)
        return [lbl, spin]

    def _choose_folder(self):
        folder = filedialog.askdirectory(title="Выберите папку с фотографиями")
        if not folder:
            return
        self.source_paths = [Path(folder)]
        self._refresh_source_text()
        self._suggest_output_from_current_sources()

    def _choose_files(self):
        files = filedialog.askopenfilenames(title="Выберите фотографии", filetypes=[("Изображения", "*.jpg *.jpeg *.png *.tif *.tiff *.webp *.bmp *.psd *.psb"), ("Все файлы", "*.*")])
        if not files:
            return
        self.source_paths = [Path(p) for p in files]
        self._refresh_source_text()
        self._suggest_output_from_current_sources()

    def _choose_output(self):
        folder = filedialog.askdirectory(title="Папка для результата", initialdir=self.output_var.get() or None)
        if folder:
            self.output_var.set(folder)
            self._output_autofilled_from_sources = False

    def _suggest_output_from_current_sources(self, *, force: bool = False) -> None:
        if not self.source_paths:
            return
        if not force and self.output_var.get().strip() and not self._output_autofilled_from_sources:
            return
        suggested = suggested_output_dir(self.source_paths)
        if suggested is not None:
            self.output_var.set(str(suggested))
            self._output_autofilled_from_sources = True

    def _clear_sources(self):
        self.source_paths = []
        self._refresh_source_text()
        self._output_autofilled_from_sources = False

    def _refresh_source_text(self):
        if not self.source_paths:
            self.sources_var.set("")
        elif len(self.source_paths) == 1:
            self.sources_var.set(str(self.source_paths[0]))
        else:
            self.sources_var.set(f"Выбрано файлов: {len(self.source_paths)} — {self.source_paths[0].name} ...")

    def _update_model_hint(self):
        key = LABEL_TO_KEY.get(self.model_label_var.get(), DEFAULT_MODEL_KEY)
        spec = get_model_spec(key)
        pieces = [
            spec.description,
            f"Лучше всего: {spec.best_for}" if spec.best_for else "",
            f"Ограничения: {spec.caveats}" if spec.caveats else "",
            f"Вход модели: {'динамический, с сохранением пропорций' if spec.preserves_aspect_ratio else f'{spec.input_size} px'}.",
            spec.speed_hint,
            spec.detail_hint,
            f"Безопасный GPU-пакет: до {spec.safe_cuda_batch}.",
            f"Лицензия: {spec.license_note}.",
        ]
        if spec.gated:
            if get_hf_token():
                pieces.append("HF_TOKEN обнаружен в окружении — будет использован автоматически; setup_bria.bat не требуется, если условия модели уже приняты.")
            else:
                pieces.append("HF_TOKEN в окружении не найден — перед первым использованием выполните setup_bria.bat.")
        self.model_hint.configure(text="\n".join(part for part in pieces if part))

    def _on_model_changed(self):
        # The model-dependent profile is intentionally "live": changing the
        # model immediately installs that model's recommended quality settings.
        if self.edge_profile_var.get() == PROFILE_RECOMMENDED:
            self._apply_edge_profile()
        self._update_model_hint()
        self._apply_context_states()

    @staticmethod
    def _set_widget_state(widget: tk.Widget, enabled: bool) -> None:
        try:
            widget.state(["!disabled"] if enabled else ["disabled"])
        except Exception:
            try:
                widget.configure(state=("normal" if enabled else "disabled"))
            except Exception:
                pass

    def _set_model_selector_enabled(self, enabled: bool) -> None:
        # readonly prevents free-form model names; disabled freezes the model for
        # the lifetime of the current operation so the visible selection always
        # matches the configuration that was actually launched.
        try:
            self.model_combo.configure(state="readonly" if enabled else "disabled")
        except tk.TclError:
            pass

    def _restore_idle_controls(self) -> None:
        self.start_button.state(["!disabled"])
        self.cancel_button.state(["disabled"])
        self.download_models_button.state(["!disabled"])
        self._set_model_selector_enabled(True)

    @staticmethod
    def _normalize_quality_settings(settings: dict) -> dict | None:
        try:
            mask = settings["mask"]
            cutout = settings["cutout"]
            normalized = {
                "mask": {
                    "black_point": float(mask["black_point"]),
                    "white_point": float(mask["white_point"]),
                    "gamma": float(mask["gamma"]),
                    "expand_pixels": int(mask["expand_pixels"]),
                    "feather_radius": float(mask["feather_radius"]),
                    "guided_refine": bool(mask["guided_refine"]),
                    "guided_max_long_edge": int(mask["guided_max_long_edge"]),
                    "guided_radius": int(mask["guided_radius"]),
                    "guided_blend": float(mask["guided_blend"]),
                },
                "cutout": {
                    "decontaminate": bool(cutout["decontaminate"]),
                    "decontam_strength": float(cutout["decontam_strength"]),
                },
            }
        except (KeyError, TypeError, ValueError, OverflowError):
            return None

        black = normalized["mask"]["black_point"]
        white = normalized["mask"]["white_point"]
        if not (0.0 <= black < white <= 1.0):
            return None
        return normalized

    @classmethod
    def _validated_custom_profiles(cls, presets: dict) -> dict[str, dict]:
        result: dict[str, dict] = {}
        if not isinstance(presets, dict):
            return result
        for name, settings in presets.items():
            if not isinstance(name, str):
                continue
            clean_name = name.strip()
            if not clean_name:
                continue
            normalized = cls._normalize_quality_settings(settings)
            if normalized is not None:
                result[clean_name] = normalized
        return result

    @staticmethod
    def _user_profile_label(name: str) -> str:
        return f"{USER_PROFILE_PREFIX}{name}"

    def _user_profile_name(self, profile: str) -> str | None:
        if not isinstance(profile, str) or not profile.startswith(USER_PROFILE_PREFIX):
            return None
        name = profile[len(USER_PROFILE_PREFIX):].strip()
        return name if name in self.custom_edge_profiles else None

    def _edge_profile_values(self) -> tuple[str, ...]:
        standard = (PROFILE_RECOMMENDED, PROFILE_NATURAL, PROFILE_CLEAN, PROFILE_NONE)
        custom = tuple(self._user_profile_label(name) for name in sorted(self.custom_edge_profiles, key=str.casefold))
        return standard + custom + (PROFILE_CUSTOM,)

    def _refresh_edge_profile_choices(self) -> None:
        values = self._edge_profile_values()
        for combo in self._edge_profile_combos:
            try:
                combo.configure(values=values)
            except tk.TclError:
                pass
        self._update_delete_preset_state()

    def _update_delete_preset_state(self) -> None:
        button = getattr(self, "delete_preset_button", None)
        if button is None:
            return
        is_user_preset = self._user_profile_name(self.edge_profile_var.get()) is not None
        self._set_widget_state(button, is_user_preset)

    def _create_custom_edge_profile(self) -> None:
        try:
            current = self._current_model_quality_settings()
        except (tk.TclError, ValueError, TypeError) as exc:
            messagebox.showerror("Пресет маски", f"Не удалось прочитать текущие параметры:\n{exc}")
            return
        current = self._normalize_quality_settings(current)
        if current is None:
            messagebox.showerror(
                "Пресет маски",
                "Текущие параметры маски некорректны. Проверьте нижний и верхний пороги и повторите сохранение.",
            )
            return

        name = simpledialog.askstring(
            "Новый пресет маски",
            "Название пользовательского пресета:",
            parent=self,
        )
        if name is None:
            return
        name = name.strip()
        if not name:
            messagebox.showwarning("Пресет маски", "Название пресета не может быть пустым.")
            return
        if len(name) > 80:
            messagebox.showwarning("Пресет маски", "Название пресета не должно быть длиннее 80 символов.")
            return
        reserved = {PROFILE_RECOMMENDED, PROFILE_NATURAL, PROFILE_CLEAN, PROFILE_NONE, PROFILE_CUSTOM}
        if name in reserved or name.startswith(USER_PROFILE_PREFIX):
            messagebox.showwarning(
                "Пресет маски",
                "Это название зарезервировано встроенным интерфейсом. Выберите другое название.",
            )
            return
        if name in self.custom_edge_profiles:
            if not messagebox.askyesno(
                "Пресет уже существует",
                f"Пользовательский пресет «{name}» уже существует. Заменить его текущими параметрами?",
            ):
                return

        old = dict(self.custom_edge_profiles)
        self.custom_edge_profiles[name] = current
        try:
            save_custom_mask_presets(self.custom_edge_profiles)
        except Exception as exc:
            self.custom_edge_profiles = old
            messagebox.showerror("Пресет маски", f"Не удалось сохранить пресет:\n{exc}")
            return

        self._refresh_edge_profile_choices()
        self.edge_profile_var.set(self._user_profile_label(name))

    def _delete_custom_edge_profile(self) -> None:
        profile = self.edge_profile_var.get()
        name = self._user_profile_name(profile)
        if name is None:
            # The button is normally disabled for built-in profiles. Keep this
            # guard so standard presets remain protected even if the method is
            # called programmatically.
            return
        if not messagebox.askyesno(
            "Удалить пользовательский пресет",
            f"Удалить пресет «{name}»?\n\nВстроенные профили при этом не изменяются.",
        ):
            return

        old = dict(self.custom_edge_profiles)
        del self.custom_edge_profiles[name]
        try:
            save_custom_mask_presets(self.custom_edge_profiles)
        except Exception as exc:
            self.custom_edge_profiles = old
            messagebox.showerror("Пресет маски", f"Не удалось удалить пресет:\n{exc}")
            return

        self._refresh_edge_profile_choices()
        self.edge_profile_var.set(PROFILE_CUSTOM)
        self._synchronize_edge_profile()

    def _edge_profile_settings(self, profile: str) -> dict | None:
        if profile == PROFILE_RECOMMENDED:
            key = LABEL_TO_KEY.get(self.model_label_var.get(), DEFAULT_MODEL_KEY)
            return model_recommended_overrides(key)
        user_name = self._user_profile_name(profile)
        if user_name is not None:
            values = self.custom_edge_profiles[user_name]
            return {
                "mask": dict(values["mask"]),
                "cutout": dict(values["cutout"]),
            }
        values = EDGE_PROFILES.get(profile)
        if not values:
            return None
        return {
            "mask": dict(values["mask"]),
            "cutout": dict(values["cutout"]),
        }

    def _apply_quality_settings(self, settings: dict) -> None:
        mask = settings["mask"]
        cutout = settings["cutout"]
        self._applying_edge_profile = True
        try:
            for var, setting_key in (
                (self.black_point_var, "black_point"),
                (self.white_point_var, "white_point"),
                (self.gamma_var, "gamma"),
                (self.expand_var, "expand_pixels"),
                (self.feather_var, "feather_radius"),
                (self.guided_var, "guided_refine"),
                (self.guided_long_var, "guided_max_long_edge"),
                (self.guided_radius_var, "guided_radius"),
                (self.guided_blend_var, "guided_blend"),
            ):
                var.set(mask[setting_key])
            self.decontaminate_var.set(bool(cutout["decontaminate"]))
            self.decontam_strength_var.set(float(cutout["decontam_strength"]))
        finally:
            self._applying_edge_profile = False

    def _apply_edge_profile(self):
        settings = self._edge_profile_settings(self.edge_profile_var.get())
        if settings is None:
            self._update_delete_preset_state()
            return
        self._apply_quality_settings(settings)
        self._apply_context_states()
        self._update_delete_preset_state()

    def _current_model_quality_settings(self) -> dict:
        return {
            "mask": {
                "black_point": float(self.black_point_var.get()),
                "white_point": float(self.white_point_var.get()),
                "gamma": float(self.gamma_var.get()),
                "expand_pixels": int(self.expand_var.get()),
                "feather_radius": float(self.feather_var.get()),
                "guided_refine": bool(self.guided_var.get()),
                "guided_max_long_edge": int(self.guided_long_var.get()),
                "guided_radius": int(self.guided_radius_var.get()),
                "guided_blend": float(self.guided_blend_var.get()),
            },
            "cutout": {
                "decontaminate": bool(self.decontaminate_var.get()),
                "decontam_strength": float(self.decontam_strength_var.get()),
            },
        }

    @staticmethod
    def _settings_equal(current: dict, recommended: dict) -> bool:
        for section in ("mask", "cutout"):
            current_section = current.get(section, {})
            recommended_section = recommended.get(section, {})
            if set(current_section) != set(recommended_section):
                return False
            for key, expected in recommended_section.items():
                actual = current_section[key]
                if isinstance(expected, bool):
                    if bool(actual) is not expected:
                        return False
                elif isinstance(expected, float):
                    if abs(float(actual) - float(expected)) > 1e-9:
                        return False
                else:
                    if actual != expected:
                        return False
        return True

    def _synchronize_edge_profile(self) -> None:
        if self._applying_edge_profile:
            return
        try:
            current = self._current_model_quality_settings()
            for profile in (PROFILE_RECOMMENDED, PROFILE_NATURAL, PROFILE_CLEAN, PROFILE_NONE):
                settings = self._edge_profile_settings(profile)
                if settings is not None and self._settings_equal(current, settings):
                    self.edge_profile_var.set(profile)
                    return
            for name in sorted(self.custom_edge_profiles, key=str.casefold):
                profile = self._user_profile_label(name)
                settings = self._edge_profile_settings(profile)
                if settings is not None and self._settings_equal(current, settings):
                    self.edge_profile_var.set(profile)
                    return
        except (tk.TclError, ValueError, TypeError):
            pass
        self.edge_profile_var.set(PROFILE_CUSTOM)

    def _on_quality_setting_changed(self) -> None:
        if self._applying_edge_profile:
            return
        self._synchronize_edge_profile()

    def _apply_context_states(self):
        guided_enabled = bool(self.guided_var.get())
        for widget in self._guided_widgets:
            self._set_widget_state(widget, guided_enabled)

        output_mode = OUTPUT_MODES.get(self.output_mode_label_var.get(), "cutout")
        cutout_enabled = output_mode in {"cutout", "both"}
        self._set_widget_state(self.decontam_check, cutout_enabled)
        decontam_enabled = cutout_enabled and bool(self.decontaminate_var.get())
        for widget in self._decontam_widgets:
            self._set_widget_state(widget, decontam_enabled)

        key = LABEL_TO_KEY.get(self.model_label_var.get(), DEFAULT_MODEL_KEY)
        spec = get_model_spec(key)
        self._set_widget_state(self.bria_setup_button, spec.gated)
        if spec.gated:
            if get_hf_token():
                self.bria_info_label.configure(
                    text="HF_TOKEN обнаружен в переменных окружения. Он будет использован автоматически. Убедитесь только, что условия BRIA приняты на Hugging Face."
                )
            else:
                self.bria_info_label.configure(
                    text="HF_TOKEN в окружении не найден. Для BRIA примите условия на Hugging Face и выполните setup_bria.bat."
                )
        else:
            self.bria_info_label.configure(
                text="Эта секция нужна только для модели BRIA. Для выбранной сейчас модели дополнительная активация не требуется."
            )

    def _setup_bria(self):
        bat = ROOT / "setup_bria.bat"
        try:
            subprocess.Popen(["cmd.exe", "/c", "start", "", str(bat)], cwd=str(ROOT))
        except Exception as exc:
            messagebox.showerror("BRIA", f"Не удалось запустить setup_bria.bat:\n{exc}")

    def _build_config(self) -> dict:
        black = float(self.black_point_var.get()); white = float(self.white_point_var.get())
        if not (0.0 <= black < white <= 1.0):
            raise ValueError("Для маски должно выполняться: 0 ≤ нижний порог < верхний порог ≤ 1.")
        key = LABEL_TO_KEY.get(self.model_label_var.get())
        if not key:
            raise ValueError("Не выбрана модель.")
        return merged_config(self.base_config, {
            "model": {"key": key},
            "files": {
                "recursive": bool(self.recursive_var.get()),
                "preserve_structure": bool(self.preserve_structure_var.get()),
                "overwrite": bool(self.overwrite_var.get()),
                "output_mode": OUTPUT_MODES[self.output_mode_label_var.get()],
                "cutout_format": FORMATS[self.format_label_var.get()],
                "cutout_suffix": self.cutout_suffix_var.get().strip() or "_cutout",
                "mask_suffix": self.mask_suffix_var.get().strip() or "_mask",
                "preserve_metadata": bool(self.preserve_metadata_var.get()),
            },
            "mask": {
                "black_point": black,
                "white_point": white,
                "gamma": float(self.gamma_var.get()),
                "expand_pixels": int(self.expand_var.get()),
                "feather_radius": float(self.feather_var.get()),
                "guided_refine": bool(self.guided_var.get()),
                "guided_max_long_edge": int(self.guided_long_var.get()),
                "guided_radius": int(self.guided_radius_var.get()),
                "guided_blend": float(self.guided_blend_var.get()),
            },
            "cutout": {
                "decontaminate": bool(self.decontaminate_var.get()),
                "decontam_strength": float(self.decontam_strength_var.get()),
            },
            "performance": {
                "device": self.device_var.get(),
                "fp16": bool(self.fp16_var.get()),
                "safe_gpu_memory": bool(self.safe_memory_var.get()),
                "gpu_batch_size": int(self.gpu_batch_var.get()),
                "prefetch_workers": int(self.prefetch_var.get()),
                "prefetch_buffer": int(self.prefetch_buffer_var.get()),
            },
        })

    def _save_state(self):
        key = LABEL_TO_KEY.get(self.model_label_var.get(), DEFAULT_MODEL_KEY)
        save_ui_state({
            "state_version": 1,
            "output": self.output_var.get(), "model_key": key,
            "output_mode": OUTPUT_MODES.get(self.output_mode_label_var.get(), "cutout"),
            "cutout_format": FORMATS.get(self.format_label_var.get(), "png"),
            "edge_profile": self.edge_profile_var.get(), "recursive": bool(self.recursive_var.get()),
            "preserve_structure": bool(self.preserve_structure_var.get()), "overwrite": bool(self.overwrite_var.get()),
            "preserve_metadata": bool(self.preserve_metadata_var.get()), "black_point": self.black_point_var.get(),
            "white_point": self.white_point_var.get(), "gamma": self.gamma_var.get(), "expand_pixels": self.expand_var.get(),
            "feather_radius": self.feather_var.get(), "guided_refine": bool(self.guided_var.get()),
            "guided_max_long_edge": self.guided_long_var.get(), "guided_radius": self.guided_radius_var.get(),
            "guided_blend": self.guided_blend_var.get(), "decontaminate": bool(self.decontaminate_var.get()),
            "decontam_strength": self.decontam_strength_var.get(), "device": self.device_var.get(), "fp16": bool(self.fp16_var.get()),
            "safe_gpu_memory": bool(self.safe_memory_var.get()), "gpu_batch_size": self.gpu_batch_var.get(),
            "prefetch_workers": self.prefetch_var.get(), "prefetch_buffer": self.prefetch_buffer_var.get(),
            "cutout_suffix": self.cutout_suffix_var.get(), "mask_suffix": self.mask_suffix_var.get(),
        })

    def _download_all_models(self):
        if self.worker and self.worker.is_alive():
            return

        total_mb = sum(spec.approximate_download_mb for spec in MODEL_SPECS.values())
        auth_note = (
            "HF_TOKEN обнаружен: он будет автоматически использован для BRIA."
            if get_hf_token()
            else "HF_TOKEN не обнаружен: BRIA может быть пропущена, если доступ к ней ещё не настроен."
        )
        if not messagebox.askyesno(
            "Скачать все модели",
            f"Загрузить все {len(MODEL_SPECS)} моделей в локальный кэш программы?\n\n"
            f"Максимальный объём — примерно {total_mb / 1024:.1f} ГБ. Уже скачанные файлы повторно не загружаются.\n"
            f"{auth_note}\n\n"
            "Скачиваются только необходимые файлы; GPU и веса моделей в RAM не загружаются.",
        ):
            return

        self.cancel_event = threading.Event()
        self.progress_var.set(0.0)
        self.percent_var.set("0.0%")
        self.status_var.set("Подготовка загрузки моделей...")
        self.detail_var.set(f"Кэш: {HF_HOME}")
        self.start_button.state(["disabled"])
        self.download_models_button.state(["disabled"])
        self.cancel_button.state(["!disabled"])
        self._set_model_selector_enabled(False)
        self.active_model_var.set("Рабочая модель не запущена — выполняется только проверка/загрузка кэша моделей")

        def progress(index, total, spec, phase):
            if phase == "start":
                pct = 100.0 * (index - 1) / max(1, total)
                msg = f"Проверка/загрузка модели {index}/{total}: {spec.label}"
            elif phase.startswith("retry:"):
                _tag, attempt, maximum = phase.split(":", 2)
                pct = 100.0 * (index - 1) / max(1, total)
                msg = f"Повтор {attempt}/{maximum}: {spec.label}"
            else:
                pct = 100.0 * index / max(1, total)
                msg = (
                    f"Готово {index}/{total}: {spec.label}"
                    if phase == "done"
                    else f"Не готова {index}/{total}: {spec.label} — проверка остальных продолжается"
                )
            self.events.put(("models_progress", (pct, msg, spec.repo_id)))

        def work():
            try:
                result = download_all_models(cancel_event=self.cancel_event, progress=progress)
                self.events.put(("models_done", result))
            except Exception as exc:
                self.events.put(("models_error", exc))

        self.worker = threading.Thread(target=work, daemon=True, name="model-download")
        self.worker.start()

    def _start(self):
        if self.worker and self.worker.is_alive():
            return
        if not self.source_paths:
            messagebox.showwarning("Нет файлов", "Выберите папку или фотографии.")
            return
        out = self.output_var.get().strip()
        if not out:
            messagebox.showwarning("Нет папки результата", "Укажите папку для результата.")
            return
        try:
            config = self._build_config()
        except Exception as exc:
            messagebox.showerror("Настройки", str(exc)); return
        self._save_state()
        self.cancel_event = threading.Event()
        self.progress_var.set(0.0); self.percent_var.set("0.0%"); self.detail_var.set("")
        selected_spec = get_model_spec(str(config["model"]["key"]))
        self.active_model_var.set(f"Выбрана для запуска: {selected_spec.label}")
        self.status_var.set("Запуск...")
        self.start_button.state(["disabled"]); self.cancel_button.state(["!disabled"])
        self.download_models_button.state(["disabled"])
        self._set_model_selector_enabled(False)
        sources = list(self.source_paths); output = Path(out)

        def work():
            try:
                pipeline = BatchPipeline(
                    config,
                    cancel_event=self.cancel_event,
                    progress=lambda p, m: self.events.put(("progress", (p, m))),
                    message=lambda m: self.events.put(("message", m)),
                    model_status=lambda phase, label: self.events.put(("model_status", (phase, label))),
                )
                stats = pipeline.run(sources, output)
                self.events.put(("done", stats))
            except CancelledError:
                self.events.put(("cancelled", None))
            except Exception as exc:
                self.events.put(("error", exc))

        self.worker = threading.Thread(target=work, daemon=True, name="background-removal")
        self.worker.start()

    def _cancel(self):
        if self.worker and self.worker.is_alive():
            self.cancel_event.set()
            if self.worker.name == "model-download":
                self.status_var.set("Отмена после текущей модели...")
            else:
                self.status_var.set("Отмена после текущего GPU-прохода...")
            self.cancel_button.state(["disabled"])

    def _poll_events(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "progress":
                    p, msg = payload
                    self.progress_var.set(float(p)); self.percent_var.set(f"{float(p):.1f}%"); self.status_var.set(str(msg))
                elif kind == "message":
                    self.detail_var.set(str(payload))
                elif kind == "model_status":
                    phase, label = payload
                    if phase == "loading":
                        self.active_model_var.set(f"Загрузка модели: {label}")
                    elif phase == "active":
                        self.active_model_var.set(f"Активная модель: {label}")
                    elif phase == "released":
                        self.active_model_var.set(f"Последняя использованная модель: {label} · выгружена после обработки")
                    elif phase == "skipped":
                        self.active_model_var.set(f"Модель не запускалась: {label} · все результаты уже существуют")
                    elif phase == "no_files":
                        self.active_model_var.set(f"Модель не запускалась: {label} · подходящие изображения не найдены")
                    elif phase == "load_failed":
                        self.active_model_var.set(f"Не удалось загрузить модель: {label}")
                elif kind == "done":
                    stats = payload
                    self._restore_idle_controls()
                    self.status_var.set("Готово"); self.progress_var.set(100.0); self.percent_var.set("100.0%")
                    self.detail_var.set(f"Обработано: {stats.files_processed} · пропущено: {stats.files_skipped} · ошибок: {stats.files_failed} · прозрачных файлов: {stats.cutouts_written} · масок: {stats.masks_written}")
                    if stats.files_processed == 0 and stats.files_skipped > 0:
                        messagebox.showinfo(
                            "Результаты не изменены",
                            "Все выходные файлы уже существуют. Новые настройки маски к ним не применялись.\n\n"
                            "Чтобы сравнить расширение/сжатие края на тех же фотографиях, включите "
                            "«Перезаписывать готовые файлы» или выберите другую папку результата.",
                        )
                elif kind == "cancelled":
                    self._restore_idle_controls()
                    self.status_var.set("Отменено пользователем")
                    if not self.active_model_var.get().startswith("Последняя использованная модель:"):
                        self.active_model_var.set("Модель не активна — обработка отменена")
                elif kind == "error":
                    self._restore_idle_controls()
                    self.status_var.set("Ошибка")
                    messagebox.showerror("Ошибка обработки", str(payload))
                elif kind == "models_progress":
                    p, msg, repo_id = payload
                    self.progress_var.set(float(p)); self.percent_var.set(f"{float(p):.1f}%")
                    self.status_var.set(str(msg)); self.detail_var.set(str(repo_id))
                elif kind == "models_done":
                    result = payload
                    self._restore_idle_controls()
                    self.active_model_var.set("Рабочая модель не запущена — завершена проверка/загрузка кэша моделей")
                    ready = len(result.ready)
                    failed = len(result.failed)
                    if result.cancelled:
                        self.status_var.set("Загрузка моделей отменена")
                        self.detail_var.set(f"Готово моделей: {ready}; ошибок до отмены: {failed}")
                    elif failed:
                        self.status_var.set("Загрузка моделей завершена с предупреждениями")
                        self.progress_var.set(100.0); self.percent_var.set("100.0%")
                        names = [get_model_spec(key).label for key in result.failed]
                        self.detail_var.set(f"Готово: {ready}/{len(MODEL_SPECS)} · не удалось: {failed}")
                        note = "\n".join(f"• {name}" for name in names)
                        bria_note = ""
                        if "bria_rmbg_2" in result.failed:
                            bria_note = "\n\nЕсли не загрузилась BRIA: проверьте HF_TOKEN и убедитесь, что условия модели приняты на Hugging Face."
                        messagebox.showwarning(
                            "Не все модели готовы",
                            f"Готово моделей: {ready}/{len(MODEL_SPECS)}.\n\n"
                            f"После автоматических повторных попыток не удалось подготовить:\n{note}{bria_note}\n\n"
                            "Остальные модели готовы к работе. Можно повторить проверку позже — уже загруженные данные будут использованы из кэша. "
                            "Подробная причина записана в background_remover_ai.log.",
                        )
                    else:
                        self.status_var.set("Все модели проверены и готовы")
                        self.progress_var.set(100.0); self.percent_var.set("100.0%")
                        self.detail_var.set(f"Все {ready} моделей готовы; необходимые файлы находятся в локальном кэше программы.")
                        messagebox.showinfo("Модели готовы", f"Все {ready} моделей проверены и готовы к работе. Уже находившиеся в кэше файлы повторно не скачивались.")
                elif kind == "models_error":
                    self._restore_idle_controls()
                    self.active_model_var.set("Рабочая модель не запущена — ошибка при проверке/загрузке кэша моделей")
                    self.status_var.set("Ошибка загрузки моделей")
                    messagebox.showerror("Загрузка моделей", str(payload))
        except queue.Empty:
            pass
        self.after(100, self._poll_events)

    def _on_close(self):
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno("Закрыть", "Операция ещё идёт. Отменить её и закрыть окно?"):
                return
            self.cancel_event.set()
        try:
            self._save_state()
        except Exception:
            pass
        self.destroy()
