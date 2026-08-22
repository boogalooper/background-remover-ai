@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "HF_HOME=%CD%\runtime\huggingface"
if exist "runtime\venv\Scripts\python.exe" (
  "runtime\venv\Scripts\python.exe" -m app.main %*
  exit /b %errorlevel%
)
echo Local Python environment is missing. Running installer...
call install.bat
if errorlevel 1 exit /b %errorlevel%
"runtime\venv\Scripts\python.exe" -m app.main %*
