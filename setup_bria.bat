@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "HF_HOME=%CD%\runtime\huggingface"

set "POWERSHELL_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if defined PROCESSOR_ARCHITEW6432 if exist "%SystemRoot%\Sysnative\WindowsPowerShell\v1.0\powershell.exe" set "POWERSHELL_EXE=%SystemRoot%\Sysnative\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%POWERSHELL_EXE%" set "POWERSHELL_EXE=powershell.exe"

if not exist "runtime\venv\Scripts\python.exe" (
  echo Run install.bat first.
  pause
  exit /b 1
)
"%POWERSHELL_EXE%" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0app\tools\repair_venv.ps1"
if errorlevel 1 (
  echo Existing Python environment cannot be used here. Run install.bat first.
  pause
  exit /b 1
)
"runtime\venv\Scripts\python.exe" -m app.tools.setup_bria
set "RC=%errorlevel%"
echo.
pause
exit /b %RC%
