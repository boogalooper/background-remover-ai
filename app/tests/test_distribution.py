from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_batch_files_windows_crlf_without_bom():
    for rel in ("install.bat", "run.bat", "setup_bria.bat"):
        data = (ROOT / rel).read_bytes()
        assert not data.startswith(b"\xef\xbb\xbf")
        assert data.startswith(b"@echo off\r\n")
        assert data.count(b"\n") == data.count(b"\r\n")


def test_installer_uses_private_managed_python():
    install = (ROOT / "install.bat").read_text(encoding="utf-8")
    lower = install.lower()
    assert "install_managed_python.ps1" in install
    assert "winget install" not in lower
    assert "py -3.11" not in lower
    assert "where python" not in lower
    assert (ROOT / "app" / "tools" / "install_managed_python.ps1").is_file()


def test_launcher_never_silently_runs_installer():
    run = (ROOT / "run.bat").read_text(encoding="utf-8")
    assert "call install.bat" not in run.lower()
    assert "repair_venv.ps1" in run
    repair = (ROOT / "app" / "tools" / "repair_venv.ps1").read_text(encoding="utf-8")
    assert "runtime\\python" in repair
    assert "ProjectDir" in repair
    assert not repair.lstrip().lower().startswith("param(")


def test_private_venv_is_relocatable_and_move_repair_preserves_packages():
    installer = (ROOT / "app" / "tools" / "install_managed_python.ps1").read_text(encoding="utf-8-sig")
    repair = (ROOT / "app" / "tools" / "repair_venv.ps1").read_text(encoding="utf-8-sig")
    assert "--relocatable" in installer
    assert "--relocatable" in repair
    assert "--allow-existing" in repair
    assert "--no-python-downloads" in repair
    assert "Move-Item -LiteralPath $tmp" not in repair


def test_relocatable_validation_does_not_require_absolute_home_metadata():
    installer = (ROOT / "app" / "tools" / "install_managed_python.ps1").read_text(encoding="utf-8-sig")
    repair = (ROOT / "app" / "tools" / "repair_venv.ps1").read_text(encoding="utf-8-sig")
    for script in (installer, repair):
        assert "sys.base_prefix" in script
        assert "APP_EXPECTED_BASE" in script
        assert "$HomeMoved" not in script
        assert "$venvHome" not in script


def test_fresh_venv_bootstraps_pip_with_private_uv():
    install = (ROOT / "install.bat").read_text(encoding="utf-8")
    assert 'set "UV=%CD%\\runtime\\uv\\uv.exe"' in install
    assert '"%UV%" pip install --python "%PY%" --upgrade pip setuptools wheel' in install
    assert "-m ensurepip" not in install
    bootstrap_prefix = install.split(":detect_gpu", 1)[0]
    assert '"%PY%" -m pip install --upgrade pip setuptools wheel' not in bootstrap_prefix
