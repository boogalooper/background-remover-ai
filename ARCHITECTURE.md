# Background Remover AI — краткая архитектура v0.1.25

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
- CPU prefetch строго ограничен `prefetch_buffer`; число одновременно работающих workers не может превышать этот буфер.
- BiRefNet HR в safe mode использует batch 1; BiRefNet Dynamic всегда использует batch 1, чтобы не смешивать разные геометрии в одном tensor batch.

## Модели

Все модели загружаются через `AutoModelForImageSegmentation(..., trust_remote_code=True)`. Для BiRefNet удалённый код закреплён на проверенных commit revisions. Обычные модели получают квадратный RGB-вход своего размера; BiRefNet Dynamic сохраняет пропорции кадра и использует динамический размер. Далее применяется ImageNet normalization, последний выход модели, sigmoid и resize маски до исходного размера.

## Выходные файлы

Исходники не изменяются. При `overwrite=false` cutout и mask проверяются независимо: если один уже есть, повторно пишется только отсутствующий.


## Relocatable private Python

`runtime\venv` создаётся через `uv venv --relocatable`. При смене пути `app/tools/repair_venv.ps1` вызывает локальный `uv venv --allow-existing --relocatable` с текущим `runtime\python`. Это регенерирует Windows launcher trampolines и `pyvenv.cfg`, не очищая `site-packages`.
