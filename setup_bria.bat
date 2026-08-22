@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "HF_HOME=%CD%\runtime\huggingface"
if not exist "runtime\venv\Scripts\python.exe" (
  echo Run install.bat first.
  pause
  exit /b 1
)
"runtime\venv\Scripts\python.exe" -m app.tools.setup_bria
set "RC=%errorlevel%"
echo.
pause
exit /b %RC%
