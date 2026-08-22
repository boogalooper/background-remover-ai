@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
title Background Remover AI launcher
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "HF_HOME=%CD%\runtime\huggingface"

echo [Background Remover AI] Starting...
echo [1/3] Checking private CPython 3.11.16 environment...

if not exist "runtime\venv\Scripts\python.exe" goto :not_installed
if not exist "app\tools\repair_venv.ps1" goto :not_installed

set "POWERSHELL_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if defined PROCESSOR_ARCHITEW6432 if exist "%SystemRoot%\Sysnative\WindowsPowerShell\v1.0\powershell.exe" set "POWERSHELL_EXE=%SystemRoot%\Sysnative\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%POWERSHELL_EXE%" set "POWERSHELL_EXE=powershell.exe"

"%POWERSHELL_EXE%" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0app\tools\repair_venv.ps1"
if errorlevel 1 goto :broken

echo [2/3] Local Python environment OK.
echo [3/3] Opening Background Remover AI...
"runtime\venv\Scripts\python.exe" -m app.main %*
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" exit /b 0

echo.
echo Background Remover AI closed with error code %RC%.
echo See background_remover_ai.log and background_remover_ai_crash.log for details.
pause
exit /b %RC%

:not_installed
echo.
echo Private Python environment is not installed yet.
echo Run install.bat once. It will download uv and its own CPython 3.11.16 x64.
echo System Python is NOT required.
pause
exit /b 1

:broken
echo.
echo The private Python environment could not be checked or repaired.
echo Run install.bat once to rebuild it. System Python is NOT required.
pause
exit /b 1
