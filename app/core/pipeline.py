from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

from PIL import Image

from app.core.config import validate_config
from app.core.scanner import common_source_root, scan_inputs
from app.image.io import LoadedImage, load_image, save_cutout, save_mask
from app.image.postprocess import process_mask
from app.models.backend import TransformersBackgroundBackend
from app.models.catalog import get_model_spec

log = logging.getLogger(__name__)
ProgressCallback = Callable[[float, str], None]
MessageCallback = Callable[[str], None]
ModelStatusCallback = Callable[[str, str], None]


class CancelledError(RuntimeError):
    pass


@dataclass
class BatchStats:
    files_found: int = 0
    files_processed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    cutouts_written: int = 0
    masks_written: int = 0


@dataclass
class OutputPaths:
    cutout: Path | None
    mask: Path | None


class BatchPipeline:
    def __init__(
        self,
        config: dict,
        *,
        cancel_event: threading.Event | None = None,
        progress: ProgressCallback | None = None,
        message: MessageCallback | None = None,
        model_status: ModelStatusCallback | None = None,
        backend_factory=None,
    ):
        self.config = config
        self.cancel_event = cancel_event or threading.Event()
        self.progress_cb = progress or (lambda _p, _m: None)
        self.message_cb = message or (lambda _m: None)
        self.model_status_cb = model_status or (lambda _phase, _label: None)
        self.backend_factory = backend_factory or TransformersBackgroundBackend
        self._last_progress = 0.0

    def _progress(self, value: float, message: str) -> None:
        value = min(100.0, max(self._last_progress, float(value)))
        self._last_progress = value
        self.progress_cb(value, message)

    def _check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise CancelledError("Cancelled")

    def _make_output_paths(self, source: Path, source_root: Path, output_root: Path) -> OutputPaths:
        files_cfg = self.config["files"]
        mode = files_cfg.get("output_mode", "cutout")
        preserve = bool(files_cfg.get("preserve_structure", True))
        try:
            rel_parent = source.parent.relative_to(source_root) if preserve else Path()
        except ValueError:
            rel_parent = Path()
        target_dir = output_root / rel_parent
        stem = source.stem
        fmt = str(files_cfg.get("cutout_format", "png")).lower()
        ext = ".tif" if fmt in {"tif", "tiff"} else ".png"
        cutout = None
        mask = None
        if mode in {"cutout", "both"}:
            cutout = target_dir / f"{stem}{files_cfg.get('cutout_suffix', '_cutout')}{ext}"
        if mode in {"mask", "both"}:
            mask = target_dir / f"{stem}{files_cfg.get('mask_suffix', '_mask')}.png"
        return OutputPaths(cutout=cutout, mask=mask)

    @staticmethod
    def _missing_outputs(paths: OutputPaths, *, overwrite: bool) -> OutputPaths:
        """Return only outputs that should actually be written.

        With overwrite disabled, an existing cutout and an existing mask are
        considered independently.  This is important for output_mode=both: if
        only one result exists, rerunning the job must create the missing file
        without silently replacing the result that is already there.
        """
        if overwrite:
            return paths
        return OutputPaths(
            cutout=None if paths.cutout is not None and paths.cutout.exists() else paths.cutout,
            mask=None if paths.mask is not None and paths.mask.exists() else paths.mask,
        )

    @staticmethod
    def _nothing_to_write(paths: OutputPaths) -> bool:
        return paths.cutout is None and paths.mask is None

    @staticmethod
    def _collision_key(path: Path) -> str:
        # The application targets Windows, whose default file systems are
        # case-insensitive.  Case-folding everywhere also keeps tests/platforms
        # conservative instead of allowing an output plan that would break only
        # after being moved to Windows.
        return str(path).replace("\\", "/").casefold()

    @classmethod
    def _validate_output_plan(cls, tasks: list[tuple[Path, OutputPaths]]) -> None:
        seen: dict[str, tuple[Path, str, Path]] = {}
        collisions: list[str] = []
        for source, outputs in tasks:
            for kind, target in (("прозрачный файл", outputs.cutout), ("маска", outputs.mask)):
                if target is None:
                    continue
                key = cls._collision_key(target)
                previous = seen.get(key)
                if previous is None:
                    seen[key] = (source, kind, target)
                    continue
                prev_source, prev_kind, prev_target = previous
                collisions.append(
                    f"{prev_target}: {prev_source.name} ({prev_kind}) ↔ {source.name} ({kind})"
                )

        if collisions:
            preview = "\n".join(f"• {line}" for line in collisions[:8])
            extra = len(collisions) - 8
            if extra > 0:
                preview += f"\n• …и ещё конфликтов: {extra}"
            raise ValueError(
                "Несколько результатов должны быть записаны в один и тот же файл. "
                "Измените суффиксы, включите сохранение структуры подпапок или выберите другую папку результата.\n\n"
                + preview
            )

    @staticmethod
    def _validate_source_output_relationship(sources: list[Path], output_root: Path) -> None:
        conflicts: list[Path] = []
        for source in sources:
            # Explicit files are safe beside their results: scan_inputs keeps
            # them even inside exclude_roots and does not recursively discover
            # the generated files.  A selected directory inside output_root is
            # unsafe because excluding output_root would hide the whole source.
            if not source.is_dir():
                continue
            try:
                source.relative_to(output_root)
            except ValueError:
                continue
            conflicts.append(source)
        if conflicts:
            preview = "\n".join(f"• {path}" for path in conflicts[:6])
            extra = len(conflicts) - 6
            if extra > 0:
                preview += f"\n• …и ещё: {extra}"
            raise ValueError(
                "Папка результата не может совпадать с исходной папкой или содержать выбранные исходники. "
                "Иначе сканер обязан исключить эти файлы, чтобы не обрабатывать собственные результаты.\n\n"
                + preview
            )

    def _prefetch(self, paths: list[Path], workers: int, max_pending: int) -> Iterator[tuple[Path, LoadedImage | Exception]]:
        max_pending = max(1, int(max_pending))
        workers = max(1, min(int(workers), max_pending))
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="image-prefetch")
        completed_normally = False
        try:
            pending: dict[int, tuple[Path, Future]] = {}
            submit_idx = 0
            yield_idx = 0
            while yield_idx < len(paths):
                self._check_cancel()
                while submit_idx < len(paths) and len(pending) < max_pending:
                    self._check_cancel()
                    path = paths[submit_idx]
                    pending[submit_idx] = (path, pool.submit(load_image, path))
                    submit_idx += 1
                path, future = pending.pop(yield_idx)
                while True:
                    self._check_cancel()
                    try:
                        value = future.result(timeout=0.1)
                        break
                    except FutureTimeoutError:
                        continue
                    except Exception as exc:
                        value = exc
                        break
                yield path, value
                yield_idx += 1
            completed_normally = True
        finally:
            if not completed_normally:
                for _path, future in pending.values():
                    future.cancel()
            pool.shutdown(wait=completed_normally, cancel_futures=not completed_normally)

    def run(self, sources: Iterable[Path], output_root: Path) -> BatchStats:
        self._last_progress = 0.0
        validate_config(self.config)
        sources = [Path(p).expanduser().resolve() for p in sources]
        output_root = Path(output_root).expanduser().resolve()
        self._validate_source_output_relationship(sources, output_root)
        self._progress(0.5, "Поиск фотографий...")
        files = scan_inputs(
            sources,
            recursive=bool(self.config["files"].get("recursive", True)),
            exclude_roots=[output_root],
        )
        stats = BatchStats(files_found=len(files))
        if not files:
            spec = get_model_spec(str(self.config["model"]["key"]))
            self.model_status_cb("no_files", spec.label)
            self._progress(100.0, "Модель не запускалась: поддерживаемые изображения не найдены")
            return stats
        source_root = common_source_root(sources) or files[0].parent
        self.message_cb(f"Найдено файлов: {len(files)}")

        # First decide what actually needs to be written.  This happens BEFORE
        # model loading so a simple re-run with overwrite disabled does not spend
        # time/VRAM loading a large model only to discover that every output exists.
        self._progress(2.0, "Проверка готовых результатов...")
        overwrite = bool(self.config["files"].get("overwrite", False))
        planned_tasks: list[tuple[Path, OutputPaths]] = [
            (path, self._make_output_paths(path, source_root, output_root)) for path in files
        ]
        self._validate_output_plan(planned_tasks)

        tasks: list[tuple[Path, OutputPaths]] = []
        for path, outputs in planned_tasks:
            outputs = self._missing_outputs(outputs, overwrite=overwrite)
            if self._nothing_to_write(outputs):
                stats.files_skipped += 1
            else:
                tasks.append((path, outputs))

        if stats.files_skipped and not overwrite:
            self.message_cb(
                f"Уже существует результатов: {stats.files_skipped}. "
                "Новые настройки маски к ним НЕ применяются, пока выключено "
                "«Перезаписывать готовые файлы»."
            )
        if not tasks:
            spec = get_model_spec(str(self.config["model"]["key"]))
            self.model_status_cb("skipped", spec.label)
            self._progress(100.0, "Модель не запускалась: все результаты уже существуют — включите перезапись, чтобы применить новые настройки")
            return stats

        mask_cfg = self.config.get("mask", {})
        cutout_cfg = self.config.get("cutout", {})
        self.message_cb(
            "Маска: "
            f"чёрный={float(mask_cfg.get('black_point', 0.0)):.2f}; "
            f"белый={float(mask_cfg.get('white_point', 1.0)):.2f}; "
            f"гамма={float(mask_cfg.get('gamma', 1.0)):.2f}; "
            f"сдвиг края={int(mask_cfg.get('expand_pixels', 0)):+d}px; "
            f"размытие={float(mask_cfg.get('feather_radius', 0.0)):.2f}px; "
            f"очистка цвета края={'вкл' if bool(cutout_cfg.get('decontaminate', True)) else 'выкл'}"
        )

        spec = get_model_spec(str(self.config["model"]["key"]))
        self.model_status_cb("loading", spec.label)
        self._progress(4.0, f"Загрузка модели: {spec.label} ({spec.repo_id}). При первом запуске модель может скачиваться...")
        backend = self.backend_factory(
            spec,
            requested_device=str(self.config["performance"].get("device", "auto")),
            fp16=bool(self.config["performance"].get("fp16", True)),
            safe_memory=bool(self.config["performance"].get("safe_gpu_memory", True)),
            batch_size=int(self.config["performance"].get("gpu_batch_size", 1)),
        )
        try:
            backend.load()
        except Exception:
            self.model_status_cb("load_failed", spec.label)
            try:
                backend.close()
            except Exception:
                log.warning("Backend cleanup after load failure failed", exc_info=True)
            raise

        self.model_status_cb("active", spec.label)
        self.message_cb(
            f"Активная модель: {spec.label}; устройство: {getattr(backend, 'device', '?')}; "
            f"GPU-пакет: {getattr(backend, 'batch_size', 1)}"
        )

        try:
            self._progress(8.0, "Подготовка файлов...")
            workers = max(1, min(8, int(self.config["performance"].get("prefetch_workers", 2))))
            pending = max(1, min(12, int(self.config["performance"].get("prefetch_buffer", workers + 1))))
            batch_size = max(1, int(getattr(backend, "batch_size", 1)))
            output_map = {path: outputs for path, outputs in tasks}
            path_list = [path for path, _ in tasks]
            loaded_batch: list[LoadedImage] = []
            path_batch: list[Path] = []
            total = len(tasks)
            completed = 0

            def flush_batch() -> None:
                nonlocal completed, loaded_batch, path_batch
                if not loaded_batch:
                    return
                self._check_cancel()
                display_name = path_batch[0].name if len(path_batch) == 1 else f"{path_batch[0].name} +{len(path_batch)-1}"
                pct0 = 10.0 + 82.0 * completed / total
                self._progress(pct0, f"Удаление фона: {completed + 1}/{total} — {display_name}")
                def predict_resilient(items: list[LoadedImage]) -> list:
                    try:
                        return backend.predict([item.image for item in items])
                    except RuntimeError as exc:
                        text = str(exc).lower()
                        is_oom = "vram" in text or "out of memory" in text or "недостаточно" in text
                        if not is_oom or len(items) <= 1:
                            raise
                        # A user may intentionally ask for a large GPU batch.  Instead of
                        # aborting the whole folder on OOM, retry the same already-loaded
                        # images as smaller batches.  This does not create another model.
                        half = max(1, len(items) // 2)
                        self.message_cb(
                            f"GPU-пакет {len(items)} не поместился в VRAM — автоматически повторяю как {half}+{len(items)-half}."
                        )
                        return predict_resilient(items[:half]) + predict_resilient(items[half:])

                masks = predict_resilient(loaded_batch)
                try:
                    masks = list(masks)
                except TypeError as exc:
                    raise RuntimeError("Backend модели вернул результат неподдерживаемого типа вместо списка масок.") from exc
                if len(masks) != len(loaded_batch):
                    raise RuntimeError(
                        "Backend модели вернул неверное число масок: "
                        f"ожидалось {len(loaded_batch)}, получено {len(masks)}. "
                        "Пакет остановлен, чтобы не потерять файлы молча."
                    )
                for loaded, mask in zip(loaded_batch, masks):
                    if not isinstance(mask, Image.Image):
                        raise RuntimeError(
                            f"Backend модели вернул маску неподдерживаемого типа: {type(mask).__name__}."
                        )
                    if mask.size != loaded.image.size:
                        raise RuntimeError(
                            "Backend модели вернул маску неверного размера: "
                            f"ожидалось {loaded.image.size[0]}×{loaded.image.size[1]}, "
                            f"получено {mask.size[0]}×{mask.size[1]}."
                        )
                for loaded, path, mask in zip(loaded_batch, path_batch, masks):
                    self._check_cancel()
                    try:
                        mask = process_mask(mask, loaded.image, self.config["mask"])
                        outputs = output_map[path]
                        if outputs.cutout is not None:
                            fmt = str(self.config["files"].get("cutout_format", "png")).upper()
                            if fmt == "TIF":
                                fmt = "TIFF"
                            save_cutout(
                                loaded,
                                mask,
                                outputs.cutout,
                                format_name=fmt,
                                preserve_metadata=bool(self.config["files"].get("preserve_metadata", True)),
                                decontaminate=bool(self.config.get("cutout", {}).get("decontaminate", True)),
                                decontam_strength=float(self.config.get("cutout", {}).get("decontam_strength", 0.5)),
                            )
                            stats.cutouts_written += 1
                        if outputs.mask is not None:
                            save_mask(mask, outputs.mask)
                            stats.masks_written += 1
                        stats.files_processed += 1
                    except Exception:
                        stats.files_failed += 1
                        log.exception("Failed to save result for %s", path)
                        self.message_cb(f"Ошибка сохранения: {path.name}")
                    completed += 1
                    pct = 10.0 + 84.0 * completed / total
                    self._progress(pct, f"Готово {completed}/{total}: {path.name}")
                loaded_batch = []
                path_batch = []

            for path, loaded_or_error in self._prefetch(path_list, workers, pending):
                self._check_cancel()
                if isinstance(loaded_or_error, Exception):
                    stats.files_failed += 1
                    completed += 1
                    log.error("Failed to load %s: %s", path, loaded_or_error)
                    self.message_cb(f"Не удалось открыть: {path.name}")
                    self._progress(10.0 + 84.0 * completed / total, f"Пропущен повреждённый файл: {path.name}")
                    continue
                loaded_batch.append(loaded_or_error)
                path_batch.append(path)
                if len(loaded_batch) >= batch_size:
                    flush_batch()
            flush_batch()
            self._progress(98.0, "Завершение и очистка памяти...")
        finally:
            try:
                backend.close()
            except Exception:
                log.warning("Backend cleanup failed", exc_info=True)
            finally:
                self.model_status_cb("released", spec.label)
        self._progress(100.0, "Готово")
        return stats
