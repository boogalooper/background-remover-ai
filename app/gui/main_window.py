from __future__ import annotations

import queue
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from app import __version__
from app.core.config import load_ui_state, merged_config, save_ui_state
from app.core.pipeline import BatchPipeline, CancelledError
from app.core.scanner import suggested_output_dir
from app.gui.scrollable import ScrollableFrame
from app.gui.tooltip import ToolTip
from app.models.catalog import LABEL_TO_KEY, MODEL_SPECS, get_model_spec
from app.paths import ROOT

MODEL_LABELS = [spec.label for spec in MODEL_SPECS.values()]
OUTPUT_MODES = {
    "PNG/TIFF с прозрачностью": "cutout",
    "Только чёрно-белая маска": "mask",
    "Прозрачный файл + маска": "both",
}
OUTPUT_MODES_INV = {v: k for k, v in OUTPUT_MODES.items()}
FORMATS = {"PNG": "png", "TIFF с альфа-каналом": "tiff"}
FORMATS_INV = {v: k for k, v in FORMATS.items()}
EDGE_PROFILES = {
    "Естественный край (рекомендуется)": {"black_point": 0.02, "white_point": 0.98, "gamma": 1.0, "expand_pixels": 0, "feather_radius": 0.0},
    "Чище фон": {"black_point": 0.05, "white_point": 0.95, "gamma": 1.0, "expand_pixels": 0, "feather_radius": 0.0},
    "Без дополнительной обработки": {"black_point": 0.0, "white_point": 1.0, "gamma": 1.0, "expand_pixels": 0, "feather_radius": 0.0},
}


class MainWindow(tk.Tk):
    def __init__(self, config: dict, initial_sources: list[str] | None = None, initial_output: str | None = None):
        super().__init__()
        self.title(f"Background Remover AI v{__version__}")
        screen_h = max(650, int(self.winfo_screenheight()))
        h = max(620, min(820, screen_h - 90))
        self.geometry(f"1080x{h}")
        self.minsize(900, min(620, h))

        self.base_config = config
        self.state = load_ui_state()
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.cancel_event = threading.Event()
        self._guided_widgets: list[tk.Widget] = []

        saved = lambda k, d: self.state.get(k, d)
        default_model_key = saved("model_key", config["model"].get("key", "bria_rmbg_2"))
        default_spec = get_model_spec(default_model_key if default_model_key in MODEL_SPECS else "bria_rmbg_2")

        self.sources_var = tk.StringVar(value="")
        self.output_var = tk.StringVar(value=saved("output", ""))
        self.model_label_var = tk.StringVar(value=default_spec.label)
        self.output_mode_label_var = tk.StringVar(value=OUTPUT_MODES_INV.get(saved("output_mode", config["files"].get("output_mode", "cutout")), "PNG/TIFF с прозрачностью"))
        self.format_label_var = tk.StringVar(value=FORMATS_INV.get(saved("cutout_format", config["files"].get("cutout_format", "png")), "PNG"))
        self.edge_profile_var = tk.StringVar(value=saved("edge_profile", "Естественный край (рекомендуется)"))
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

        self.status_var = tk.StringVar(value="Выберите папку или фотографии")
        self.detail_var = tk.StringVar(value="")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.percent_var = tk.StringVar(value="0.0%")

        self._build_ui()
        self._refresh_source_text()
        self._apply_context_states()
        self.after(100, self._poll_events)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)
        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True)

        quick_scroll = ScrollableFrame(notebook)
        advanced_scroll = ScrollableFrame(notebook)
        notebook.add(quick_scroll, text="Быстрый старт")
        notebook.add(advanced_scroll, text="Расширенные")
        quick = quick_scroll.inner
        advanced = advanced_scroll.inner
        quick.columnconfigure(1, weight=1)
        advanced.columnconfigure(1, weight=1)

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
        model_combo = ttk.Combobox(settings, textvariable=self.model_label_var, values=MODEL_LABELS, state="readonly", width=48)
        model_combo.grid(row=0, column=1, sticky="ew", pady=4)
        model_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_model_hint())
        self.model_hint = ttk.Label(settings, text="", wraplength=760)
        self.model_hint.grid(row=1, column=1, sticky="w", pady=(0, 6))
        self._update_model_hint()

        ttk.Label(settings, text="Выход:").grid(row=2, column=0, sticky="w", padx=(0, 10), pady=4)
        ttk.Combobox(settings, textvariable=self.output_mode_label_var, values=list(OUTPUT_MODES), state="readonly").grid(row=2, column=1, sticky="ew", pady=4)
        ttk.Label(settings, text="Формат прозрачного файла:").grid(row=3, column=0, sticky="w", padx=(0, 10), pady=4)
        ttk.Combobox(settings, textvariable=self.format_label_var, values=list(FORMATS), state="readonly").grid(row=3, column=1, sticky="ew", pady=4)
        ttk.Label(settings, text="Край:").grid(row=4, column=0, sticky="w", padx=(0, 10), pady=4)
        edge = ttk.Combobox(settings, textvariable=self.edge_profile_var, values=list(EDGE_PROFILES), state="readonly")
        edge.grid(row=4, column=1, sticky="ew", pady=4)
        edge.bind("<<ComboboxSelected>>", lambda _e: self._apply_edge_profile())

        buttons = ttk.Frame(quick)
        buttons.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(14, 4)); r += 1
        buttons.columnconfigure(0, weight=1)
        self.start_button = ttk.Button(buttons, text="НАЧАТЬ ОБРАБОТКУ", command=self._start)
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 8), ipady=8)
        self.cancel_button = ttk.Button(buttons, text="Отмена", command=self._cancel, state="disabled")
        self.cancel_button.grid(row=0, column=1, sticky="e", ipady=8)

        progress_box = ttk.LabelFrame(quick, text="Ход работы", padding=10)
        progress_box.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(5, 10)); progress_box.columnconfigure(0, weight=1); r += 1
        line = ttk.Frame(progress_box); line.grid(row=0, column=0, sticky="ew"); line.columnconfigure(0, weight=1)
        ttk.Label(line, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        ttk.Label(line, textvariable=self.percent_var).grid(row=0, column=1, sticky="e")
        ttk.Progressbar(progress_box, variable=self.progress_var, maximum=100).grid(row=1, column=0, sticky="ew", pady=(6, 4))
        ttk.Label(progress_box, textvariable=self.detail_var, wraplength=940).grid(row=2, column=0, sticky="w")

        # Advanced: mask
        r = 0
        mask_box = ttk.LabelFrame(advanced, text="Маска и край", padding=10)
        mask_box.grid(row=r, column=0, columnspan=3, sticky="ew", pady=5); mask_box.columnconfigure(1, weight=1); r += 1
        self._spinrow(mask_box, 0, "Удалять слабый фон ниже:", self.black_point_var, 0.0, 0.5, 0.01, "Чем выше, тем сильнее очищаются полупрозрачные остатки фона.")
        self._spinrow(mask_box, 1, "Считать объектом выше:", self.white_point_var, 0.5, 1.0, 0.01, "Чем ниже, тем быстрее полупрозрачные участки становятся непрозрачными.")
        self._spinrow(mask_box, 2, "Гамма маски:", self.gamma_var, 0.3, 3.0, 0.05, "1.0 — без изменения. Используйте только для тонкой настройки края.")
        self._spinrow(mask_box, 3, "Сдвиг края маски, px (итоговый файл):", self.expand_var, -100, 100, 1, "Положительное значение расширяет объект, отрицательное — сжимает. Значение измеряется в пикселях итогового полноразмерного изображения; на 30–60 Мп фото 2–5 px почти незаметны, для проверки попробуйте ±20…40 px.")
        self._spinrow(mask_box, 4, "Размытие края, px:", self.feather_var, 0.0, 10.0, 0.25, "Обычно лучше оставить 0: современные модели уже дают мягкий край.")
        gcheck = ttk.Checkbutton(mask_box, text="Экспериментально уточнять край по исходному изображению", variable=self.guided_var, command=self._apply_context_states)
        gcheck.grid(row=5, column=0, columnspan=2, sticky="w", pady=(8, 3))
        self._guided_widgets += self._spinrow(mask_box, 6, "Макс. размер уточнения, px:", self.guided_long_var, 1024, 8192, 256, "Ограничивает расход RAM при работе с 40–60 Мп файлами.")
        self._guided_widgets += self._spinrow(mask_box, 7, "Радиус уточнения:", self.guided_radius_var, 1, 32, 1, "Больший радиус сильнее привязывает маску к крупным границам изображения.")
        self._guided_widgets += self._spinrow(mask_box, 8, "Сила уточнения:", self.guided_blend_var, 0.0, 1.0, 0.05, "0 — выключено, 1 — максимальное влияние. Рекомендуется около 0.35.")

        perf = ttk.LabelFrame(advanced, text="Производительность и память", padding=10)
        perf.grid(row=r, column=0, columnspan=3, sticky="ew", pady=5); perf.columnconfigure(1, weight=1); r += 1
        ttk.Label(perf, text="Устройство:").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Combobox(perf, textvariable=self.device_var, values=("auto", "cuda", "cpu"), state="readonly", width=18).grid(row=0, column=1, sticky="w", pady=4)
        ttk.Checkbutton(perf, text="FP16 на NVIDIA (быстрее и экономнее VRAM)", variable=self.fp16_var).grid(row=1, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Checkbutton(perf, text="Безопасно ограничивать GPU-пакет для выбранной модели", variable=self.safe_memory_var).grid(row=2, column=0, columnspan=2, sticky="w", pady=4)
        self._spinrow(perf, 3, "Кадров за один GPU-проход:", self.gpu_batch_var, 1, 8, 1, "Это один экземпляр модели с батчем, а не несколько копий модели в VRAM.")
        self._spinrow(perf, 4, "Потоки чтения файлов:", self.prefetch_var, 1, 8, 1, "Пока GPU считает текущий пакет, CPU заранее читает следующие изображения.")
        self._spinrow(perf, 5, "Буфер предзагрузки:", self.prefetch_buffer_var, 1, 12, 1, "Большой буфер расходует больше RAM. Для очень больших фото 3–4 обычно достаточно.")

        files_box = ttk.LabelFrame(advanced, text="Файлы", padding=10)
        files_box.grid(row=r, column=0, columnspan=3, sticky="ew", pady=5); files_box.columnconfigure(1, weight=1); r += 1
        ttk.Checkbutton(files_box, text="Сохранять доступные ICC/EXIF/DPI в результате", variable=self.preserve_metadata_var).grid(row=0, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Label(files_box, text="Суффикс прозрачного файла:").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(files_box, textvariable=self.cutout_suffix_var, width=20).grid(row=1, column=1, sticky="w", pady=4)
        ttk.Label(files_box, text="Суффикс маски:").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(files_box, textvariable=self.mask_suffix_var, width=20).grid(row=2, column=1, sticky="w", pady=4)

        bria = ttk.LabelFrame(advanced, text="BRIA RMBG-2.0", padding=10)
        bria.grid(row=r, column=0, columnspan=3, sticky="ew", pady=5); r += 1
        ttk.Label(bria, text="BRIA требует один раз принять условия некоммерческого использования на Hugging Face.", wraplength=830).grid(row=0, column=0, sticky="w")
        ttk.Button(bria, text="Настроить доступ к BRIA...", command=self._setup_bria).grid(row=1, column=0, sticky="w", pady=(8, 0))

        ttk.Button(advanced, text="Вернуть рекомендуемые значения", command=self._reset_recommended).grid(row=r, column=0, sticky="w", pady=(10, 20))

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
        files = filedialog.askopenfilenames(title="Выберите фотографии", filetypes=[("Изображения", "*.jpg *.jpeg *.png *.tif *.tiff *.webp *.bmp"), ("Все файлы", "*.*")])
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
        key = LABEL_TO_KEY.get(self.model_label_var.get(), "bria_rmbg_2")
        spec = get_model_spec(key)
        text = f"{spec.description}  Вход модели: {spec.input_size} px. {spec.license_note}."
        if spec.gated:
            text += " Перед первым использованием выполните setup_bria.bat."
        self.model_hint.configure(text=text)

    def _apply_edge_profile(self):
        values = EDGE_PROFILES.get(self.edge_profile_var.get())
        if not values:
            return
        self.black_point_var.set(values["black_point"])
        self.white_point_var.set(values["white_point"])
        self.gamma_var.set(values["gamma"])
        self.expand_var.set(values["expand_pixels"])
        self.feather_var.set(values["feather_radius"])

    def _apply_context_states(self):
        enabled = bool(self.guided_var.get())
        for widget in self._guided_widgets:
            try:
                widget.state(["!disabled"] if enabled else ["disabled"])
            except Exception:
                pass

    def _setup_bria(self):
        bat = ROOT / "setup_bria.bat"
        try:
            subprocess.Popen(["cmd.exe", "/c", "start", "", str(bat)], cwd=str(ROOT))
        except Exception as exc:
            messagebox.showerror("BRIA", f"Не удалось запустить setup_bria.bat:\n{exc}")

    def _reset_recommended(self):
        cfg = self.base_config
        self.model_label_var.set(get_model_spec(cfg["model"]["key"]).label)
        self.output_mode_label_var.set(OUTPUT_MODES_INV[cfg["files"]["output_mode"]])
        self.format_label_var.set(FORMATS_INV[cfg["files"]["cutout_format"]])
        self.edge_profile_var.set("Естественный край (рекомендуется)")
        self.recursive_var.set(bool(cfg["files"]["recursive"]))
        self.preserve_structure_var.set(bool(cfg["files"]["preserve_structure"]))
        self.overwrite_var.set(bool(cfg["files"]["overwrite"]))
        self.preserve_metadata_var.set(bool(cfg["files"]["preserve_metadata"]))
        for var, key in ((self.black_point_var, "black_point"), (self.white_point_var, "white_point"), (self.gamma_var, "gamma"), (self.expand_var, "expand_pixels"), (self.feather_var, "feather_radius"), (self.guided_var, "guided_refine"), (self.guided_long_var, "guided_max_long_edge"), (self.guided_radius_var, "guided_radius"), (self.guided_blend_var, "guided_blend")):
            var.set(cfg["mask"][key])
        self.device_var.set(cfg["performance"]["device"])
        self.fp16_var.set(cfg["performance"]["fp16"])
        self.safe_memory_var.set(cfg["performance"]["safe_gpu_memory"])
        self.gpu_batch_var.set(cfg["performance"]["gpu_batch_size"])
        self.prefetch_var.set(cfg["performance"]["prefetch_workers"])
        self.prefetch_buffer_var.set(cfg["performance"]["prefetch_buffer"])
        self.cutout_suffix_var.set(cfg["files"]["cutout_suffix"])
        self.mask_suffix_var.set(cfg["files"]["mask_suffix"])
        self._update_model_hint(); self._apply_context_states()

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
        key = LABEL_TO_KEY.get(self.model_label_var.get(), "bria_rmbg_2")
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
            "guided_blend": self.guided_blend_var.get(), "device": self.device_var.get(), "fp16": bool(self.fp16_var.get()),
            "safe_gpu_memory": bool(self.safe_memory_var.get()), "gpu_batch_size": self.gpu_batch_var.get(),
            "prefetch_workers": self.prefetch_var.get(), "prefetch_buffer": self.prefetch_buffer_var.get(),
            "cutout_suffix": self.cutout_suffix_var.get(), "mask_suffix": self.mask_suffix_var.get(),
        })

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
        self.status_var.set("Запуск...")
        self.start_button.state(["disabled"]); self.cancel_button.state(["!disabled"])
        sources = list(self.source_paths); output = Path(out)

        def work():
            try:
                pipeline = BatchPipeline(
                    config,
                    cancel_event=self.cancel_event,
                    progress=lambda p, m: self.events.put(("progress", (p, m))),
                    message=lambda m: self.events.put(("message", m)),
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
                elif kind == "done":
                    stats = payload
                    self.start_button.state(["!disabled"]); self.cancel_button.state(["disabled"])
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
                    self.start_button.state(["!disabled"]); self.cancel_button.state(["disabled"]); self.status_var.set("Отменено пользователем")
                elif kind == "error":
                    self.start_button.state(["!disabled"]); self.cancel_button.state(["disabled"]); self.status_var.set("Ошибка")
                    messagebox.showerror("Ошибка обработки", str(payload))
        except queue.Empty:
            pass
        self.after(100, self._poll_events)

    def _on_close(self):
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno("Закрыть", "Обработка ещё идёт. Отменить её и закрыть окно?"):
                return
            self.cancel_event.set()
        try:
            self._save_state()
        except Exception:
            pass
        self.destroy()
