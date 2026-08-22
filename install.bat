@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "HF_HOME=%CD%\runtime\huggingface"
set "BGREMOVER_PIP_INSECURE=0"
set "NVIDIA_DETECTED=0"
set "UV_SYSTEM_CERTS=true"

if not exist "runtime" mkdir "runtime"
if not exist "runtime\huggingface" mkdir "runtime\huggingface"

echo ==============================================
echo Background Remover AI - installation v0.1.12
echo Private Python: CPython 3.11.16 x64 via uv
echo System Python and winget are not used.
echo ==============================================
echo.
echo Connection mode:
echo   [1] Normal secure mode ^(recommended^)
echo   [2] Compatibility mode for HTTPS interception ^(Kaspersky etc.^)
echo   [3] Cancel
choice /C 123 /N /M "Choose 1, 2 or 3: "
if errorlevel 3 goto :cancelled
if errorlevel 2 set "BGREMOVER_PIP_INSECURE=1"

set "POWERSHELL_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if defined PROCESSOR_ARCHITEW6432 if exist "%SystemRoot%\Sysnative\WindowsPowerShell\v1.0\powershell.exe" set "POWERSHELL_EXE=%SystemRoot%\Sysnative\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%POWERSHELL_EXE%" set "POWERSHELL_EXE=powershell.exe"

if not exist "app\tools\install_managed_python.ps1" (
  echo ERROR: app\tools\install_managed_python.ps1 is missing.
  goto :failed
)
if not exist "app\tools\repair_venv.ps1" (
  echo ERROR: app\tools\repair_venv.ps1 is missing.
  goto :failed
)

echo.
echo Preparing private Python. No system Python is required...
"%POWERSHELL_EXE%" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0app\tools\install_managed_python.ps1"
if errorlevel 1 goto :failed

set "PY=%CD%\runtime\venv\Scripts\python.exe"
set "UV=%CD%\runtime\uv\uv.exe"
if not exist "%PY%" goto :failed
if not exist "%UV%" goto :failed

echo.
echo Bootstrapping pip/setuptools/wheel with private uv...
if "%BGREMOVER_PIP_INSECURE%"=="1" goto :pip_bootstrap_compat
"%UV%" pip install --python "%PY%" --upgrade pip setuptools wheel
if errorlevel 1 goto :failed
goto :detect_gpu

:pip_bootstrap_compat
"%UV%" pip install --python "%PY%" --upgrade pip setuptools wheel --allow-insecure-host pypi.org --allow-insecure-host files.pythonhosted.org
if errorlevel 1 goto :failed

:detect_gpu
echo.
echo Detecting NVIDIA with 64-bit Python...
"%PY%" -m app.tools.detect_nvidia
if errorlevel 1 goto :install_cpu
set "NVIDIA_DETECTED=1"
goto :install_cuda

:install_cuda
echo NVIDIA detected. Checking PyTorch CUDA support...
"%PY%" -c "import torch,sys; v=torch.__version__.split('+')[0]; ok=(v=='2.9.1' and torch.version.cuda is not None and torch.cuda.is_available()); sys.exit(0 if ok else 1)" >nul 2>nul
if not errorlevel 1 goto :cuda_ready

echo Existing PyTorch is missing, CPU-only, or incompatible.
echo Replacing it with PyTorch 2.9.1 CUDA 12.8...
"%PY%" -m pip uninstall -y torch torchvision torchaudio >nul 2>nul
if "%BGREMOVER_PIP_INSECURE%"=="1" goto :install_cuda_compat
"%PY%" -m pip install --upgrade torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :failed
goto :cuda_ready

:install_cuda_compat
"%PY%" -m pip install --upgrade torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128 --trusted-host download.pytorch.org
if errorlevel 1 goto :failed

:cuda_ready
echo CUDA PyTorch is installed.
goto :deps

:install_cpu
set "NVIDIA_DETECTED=0"
set "UV_SYSTEM_CERTS=true"
echo NVIDIA CUDA driver not detected by 64-bit Python.
echo Checking CPU-capable PyTorch...
"%PY%" -c "import torch,sys; v=torch.__version__.split('+')[0]; sys.exit(0 if v=='2.9.1' else 1)" >nul 2>nul
if not errorlevel 1 goto :deps
echo Installing CPU PyTorch 2.9.1...
if "%BGREMOVER_PIP_INSECURE%"=="1" goto :install_cpu_compat
"%PY%" -m pip install --upgrade torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto :failed
goto :deps

:install_cpu_compat
"%PY%" -m pip install --upgrade torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu --trusted-host download.pytorch.org
if errorlevel 1 goto :failed

:deps
echo.
echo Installing application dependencies...
if "%BGREMOVER_PIP_INSECURE%"=="1" goto :deps_compat
"%PY%" -m pip install -r "app\requirements.txt"
if errorlevel 1 goto :failed
goto :compile

:deps_compat
"%PY%" -m pip install -r "app\requirements.txt" --trusted-host pypi.org --trusted-host files.pythonhosted.org
if errorlevel 1 goto :failed

:compile
"%PY%" -m compileall -q app
if errorlevel 1 goto :failed

echo.
echo Running GPU self-test...
if "%NVIDIA_DETECTED%"=="1" goto :selftest_cuda
"%PY%" -m app.tools.cuda_selftest
if errorlevel 1 goto :failed
goto :success

:selftest_cuda
"%PY%" -m app.tools.cuda_selftest --require-cuda
if errorlevel 1 goto :failed

:success
echo.
echo ==============================================
echo Installation complete - private CPython 3.11.16 environment ready.
echo ==============================================
if "%NVIDIA_DETECTED%"=="1" echo NVIDIA CUDA mode is ready.
if not "%NVIDIA_DETECTED%"=="1" echo CPU mode is ready.
echo Models are downloaded only when first used.
echo BiRefNet models require no account.
echo BRIA RMBG-2.0 requires one-time setup via setup_bria.bat.
echo.
echo Start with run.bat
pause
exit /b 0

:cancelled
echo.
echo Installation cancelled.
pause
exit /b 1

:failed
echo.
echo Installation failed.
echo If NVIDIA is installed, the lines above must show CUDA wheel and CUDA available: True.
pause
exit /b 1
