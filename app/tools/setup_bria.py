from __future__ import annotations

import getpass
import sys
import webbrowser

from app.paths import configure_runtime_environment

MODEL_URL = "https://huggingface.co/briaai/RMBG-2.0"


def main() -> int:
    configure_runtime_environment()
    print("BRIA RMBG-2.0 — настройка личного/некоммерческого доступа")
    print("1. В браузере войдите в Hugging Face и примите условия модели.")
    print("2. Создайте Read token в настройках Hugging Face.")
    print("3. Вернитесь сюда и вставьте токен. Он сохранится только в runtime\\huggingface этой программы.")
    print()
    try:
        webbrowser.open(MODEL_URL)
    except Exception:
        pass
    input("Нажмите Enter, когда условия приняты...")
    token = getpass.getpass("Hugging Face token (ввод скрыт): ").strip()
    if not token:
        print("Токен не введён.")
        return 2
    try:
        from huggingface_hub import login, snapshot_download
        login(token=token, add_to_git_credential=False)
        print("Проверяю доступ и скачиваю модель в локальный кэш...")
        snapshot_download("briaai/RMBG-2.0")
    except Exception as exc:
        print(f"ОШИБКА: {exc}")
        return 1
    print("Готово. BRIA RMBG-2.0 доступна приложению.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
