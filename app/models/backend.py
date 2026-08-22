from __future__ import annotations

import gc
import logging
import os
import ctypes
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from app.models.catalog import ModelSpec, resolve_batch_size
from app.paths import configure_runtime_environment

log = logging.getLogger(__name__)

MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


class ModelLoadError(RuntimeError):
    pass


def _windows_nvidia_driver_present() -> bool:
    """Detect a Windows NVIDIA CUDA driver without relying on nvidia-smi/PATH.

    This is deliberately independent of the bitness of the process that launched
    the application.  It lets us distinguish "user intentionally chose CPU"
    from "installer accidentally installed a CPU-only PyTorch build".
    """
    if os.name != "nt":
        return False
    try:
        ctypes.WinDLL("nvcuda.dll")  # type: ignore[attr-defined]
        return True
    except OSError:
        return False


def _set_model_precision(model, *, use_fp16: bool):
    """Put every floating parameter/buffer into the same precision as input.

    Some remote-code segmentation checkpoints are stored as FP16 weights even
    when loaded on CPU.  model.to("cpu") does not change dtype, which can lead
    to `Input type (float) and bias type (Half) should be the same`.
    """
    if use_fp16:
        return model.half()
    return model.float()


class TransformersBackgroundBackend:
    """One model instance, optionally processing several images as one GPU batch.

    This intentionally avoids multiple independent CUDA model sessions.  It is
    much friendlier to VRAM than launching one full model per worker.
    """

    def __init__(
        self,
        spec: ModelSpec,
        *,
        requested_device: str = "auto",
        fp16: bool = True,
        safe_memory: bool = True,
        batch_size: int = 1,
    ):
        configure_runtime_environment()
        self.spec = spec
        self.requested_device = requested_device
        self.fp16_requested = bool(fp16)
        self.safe_memory = bool(safe_memory)
        self.requested_batch_size = int(batch_size)
        self.model = None
        self.torch = None
        self.device = "cpu"
        self.fp16 = False
        self.batch_size = 1

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForImageSegmentation
        except Exception as exc:
            raise ModelLoadError(
                "Не найдены библиотеки PyTorch/Transformers. Запустите install.bat повторно."
            ) from exc

        self.torch = torch
        if self.requested_device == "cpu":
            self.device = "cpu"
        elif self.requested_device == "cuda":
            if not torch.cuda.is_available():
                raise ModelLoadError("CUDA выбрана вручную, но PyTorch не видит NVIDIA GPU.")
            self.device = "cuda"
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                if _windows_nvidia_driver_present():
                    raise ModelLoadError(
                        "В системе обнаружена NVIDIA, но установленный PyTorch не поддерживает CUDA. "
                        "Это может произойти, если install.bat был запущен из 32-битного файлового менеджера. "
                        "Запустите новый install.bat повторно — он заменит CPU-сборку PyTorch на CUDA-сборку."
                    )
                self.device = "cpu"

        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        self.fp16 = bool(self.fp16_requested and self.device == "cuda")
        self.batch_size = resolve_batch_size(
            self.requested_batch_size, self.spec, self.device, self.safe_memory
        )

        log.info(
            "Loading model %s | device=%s fp16=%s batch=%d safe_memory=%s",
            self.spec.repo_id,
            self.device,
            self.fp16,
            self.batch_size,
            self.safe_memory,
        )
        try:
            model = AutoModelForImageSegmentation.from_pretrained(
                self.spec.repo_id,
                trust_remote_code=True,
            )
        except Exception as exc:
            msg = str(exc)
            if self.spec.gated:
                raise ModelLoadError(
                    "Не удалось открыть BRIA RMBG-2.0. Эта модель требует один раз принять "
                    "условия на Hugging Face и выполнить setup_bria.bat. "
                    f"Исходная ошибка: {msg}"
                ) from exc
            raise ModelLoadError(f"Не удалось загрузить модель {self.spec.repo_id}: {msg}") from exc

        model.eval()
        model.to(self.device)
        if self.fp16:
            try:
                _set_model_precision(model, use_fp16=True)
            except Exception:
                log.warning("FP16 model conversion failed; continuing in FP32", exc_info=True)
                self.fp16 = False
                _set_model_precision(model, use_fp16=False)
        else:
            # Important for CPU and explicit FP32 CUDA: model repositories may
            # store checkpoint weights in FP16.  Moving devices alone does not
            # cast them back to FP32.
            _set_model_precision(model, use_fp16=False)
        self.model = model

    def _prepare(self, image: Image.Image):
        assert self.torch is not None
        rgb = image.convert("RGB")
        resized = rgb.resize((self.spec.input_size, self.spec.input_size), Image.Resampling.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        arr = (arr - MEAN) / STD
        tensor = self.torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        if self.fp16:
            tensor = tensor.half()
        return tensor

    def predict(self, images: Sequence[Image.Image]) -> list[Image.Image]:
        if self.model is None or self.torch is None:
            raise RuntimeError("Model is not loaded")
        if not images:
            return []
        torch = self.torch
        dtype = torch.float16 if self.fp16 else torch.float32
        batch = torch.stack([self._prepare(image) for image in images], dim=0).to(
            device=self.device, dtype=dtype
        )
        try:
            with torch.inference_mode():
                output = self.model(batch)
                preds = output[-1] if isinstance(output, (list, tuple)) else output
                preds = preds.sigmoid().float().cpu()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and self.device == "cuda":
                try:
                    del batch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                raise RuntimeError(
                    "Недостаточно VRAM для выбранного GPU-пакета. Уменьшите «Кадров за один GPU-проход» "
                    "или включите безопасный режим памяти."
                ) from exc
            raise
        finally:
            try:
                del batch
            except Exception:
                pass

        result: list[Image.Image] = []
        for idx, image in enumerate(images):
            pred = preds[idx].squeeze().numpy()
            pred = np.clip(pred * 255.0 + 0.5, 0, 255).astype(np.uint8)
            mask = Image.fromarray(pred, mode="L").resize(image.size, Image.Resampling.LANCZOS)
            result.append(mask)
        return result

    def close(self) -> None:
        model = self.model
        self.model = None
        if model is not None:
            try:
                model.to("cpu")
            except Exception:
                pass
            del model
        gc.collect()
        if self.torch is not None and self.device == "cuda":
            try:
                self.torch.cuda.empty_cache()
                self.torch.cuda.ipc_collect()
            except Exception:
                pass
