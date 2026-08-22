# Background Remover AI — краткая архитектура v0.1.12

Этот файл предназначен для дальнейшей разработки. Пользовательская инструкция находится в `README.md`.

## Поток обработки

1. `scanner.py` собирает поддерживаемые файлы и исключает папку результата.
2. `pipeline.py` создаёт один экземпляр выбранной модели.
3. CPU-пул заранее читает ограниченное число изображений, сохраняя исходный порядок.
4. `backend.py` формирует GPU-batch для одной модели. Несколько независимых копий модели не создаются.
5. Модель возвращает grayscale alpha matte, который масштабируется до исходного размера.
6. `postprocess.py` применяет выбранную коррекцию края; guided refinement опционален.
7. `io.py` сохраняет RGBA PNG/TIFF и/или mask PNG атомарно.
8. Модель выгружается, запускается GC и очистка CUDA cache.

## Защита памяти

- `safe_gpu_memory` ограничивает batch согласно каталогу модели.
- При CUDA OOM пакет рекурсивно делится на меньшие без создания новой модели.
- CPU prefetch ограничен количеством workers и buffer.
- BiRefNet HR в safe mode использует batch 1.

## Модели

Все модели загружаются через `AutoModelForImageSegmentation(..., trust_remote_code=True)` и имеют общий путь инференса: fixed-size RGB resize, ImageNet normalization, последний выход модели, sigmoid, resize mask до исходного размера.

## Выходные файлы

Исходники не изменяются. При `overwrite=false` cutout и mask проверяются независимо: если один уже есть, повторно пишется только отсутствующий.


## Relocatable private Python

`runtime\venv` создаётся через `uv venv --relocatable`. При смене пути `app/tools/repair_venv.ps1` вызывает локальный `uv venv --allow-existing --relocatable` с текущим `runtime\python`. Это регенерирует Windows launcher trampolines и `pyvenv.cfg`, не очищая `site-packages`.
