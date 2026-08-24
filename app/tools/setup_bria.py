from __future__ import annotations

import getpass
import webbrowser

from app.paths import configure_runtime_environment, get_hf_token

MODEL_ID = "briaai/RMBG-2.0"
MODEL_URL = "https://huggingface.co/briaai/RMBG-2.0"


def _verify_and_download(token: str | None) -> None:
    from huggingface_hub import snapshot_download

    print("Проверяю доступ и скачиваю модель в локальный кэш...")
    snapshot_download(MODEL_ID, token=token)


def main() -> int:
    configure_runtime_environment()
    print("BRIA RMBG-2.0 — настройка личного/некоммерческого доступа")
    print()

    env_token = get_hf_token()
    if env_token:
        print("Найден HF_TOKEN в переменных окружения Windows.")
        print("Использую его автоматически; токен не будет показан или сохранён программой повторно.")
        print()
        try:
            _verify_and_download(env_token)
        except Exception as exc:
            print(f"ОШИБКА: {exc}")
            print()
            print("Проверьте, что токен действителен и условия BRIA RMBG-2.0 приняты на Hugging Face.")
            return 1
        print("Готово. BRIA RMBG-2.0 доступна приложению через HF_TOKEN.")
        return 0

    print("HF_TOKEN в окружении не найден.")
    print("1. В браузере войдите в Hugging Face и примите условия модели.")
    print("2. Создайте Read token в настройках Hugging Face.")
    print("3. Вернитесь сюда и вставьте токен. Он будет сохранён только в runtime\\huggingface этой программы.")
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
        from huggingface_hub import login

        # Only the manual fallback persists a token in the app-local HF_HOME.
        login(token=token, add_to_git_credential=False)
        _verify_and_download(token)
    except Exception as exc:
        print(f"ОШИБКА: {exc}")
        return 1
    print("Готово. BRIA RMBG-2.0 доступна приложению.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
