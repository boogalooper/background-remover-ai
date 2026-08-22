@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "HF_HOME=%CD%\runtime\huggingface"
set "BGREMOVER_PIP_INSECURE=0"
set "BASEPY="
set "PY_PATH_FILE=runtime\python311_path.tmp"
set "NVIDIA_DETECTED=0"

if not exist "runtime" mkdir "runtime"
if not exist "runtime\huggingface" mkdir "runtime\huggingface"

echo ==============================================
echo Background Remover AI - installation v0.1.5
echo ==============================================
echo.
echo Connection mode:
echo   [1] Normal secure mode ^(recommended^)
echo   [2] Compatibility mode for HTTPS interception ^(Kaspersky etc.^)
echo   [3] Cancel
choice /C 123 /N /M "Choose 1, 2 or 3: "
if errorlevel 3 goto :cancelled
if errorlevel 2 set "BGREMOVER_PIP_INSECURE=1"

rem Reuse only a healthy local 64-bit Python 3.11 virtual environment.
if not exist "runtime\venv\Scripts\python.exe" goto :find_python
"runtime\venv\Scripts\python.exe" -c "import sys,pathlib; expected=pathlib.Path('runtime/venv').resolve(); actual=pathlib.Path(sys.prefix).resolve(); assert sys.version_info[:2]==(3,11); assert sys.maxsize > 2**32; assert actual==expected" >nul 2>nul
if not errorlevel 1 goto :have_venv
echo Existing venv is invalid here. Recreating...
rmdir /s /q "runtime\venv"

:find_python
call :detect_python311
if defined BASEPY goto :create_venv

where winget >nul 2>nul
if errorlevel 1 goto :python_missing

echo.
echo Python 3.11 x64 was not found. Installing it with winget...
winget install --id Python.Python.3.11 -e --silent --accept-package-agreements --accept-source-agreements
if errorlevel 1 goto :failed

rem The launcher/PATH in this already-open cmd.exe may not refresh after winget.
call :detect_python311
if defined BASEPY goto :create_venv
call :detect_known_python311_paths
if defined BASEPY goto :create_venv

echo.
echo Python 3.11 was installed, but python.exe could not be located in this cmd session.
echo Close this window and run install.bat again.
goto :failed

:python_missing
echo.
echo Python 3.11 x64 was not found and winget is unavailable.
echo Install Python 3.11 x64 and run install.bat again.
goto :failed

:create_venv
echo.
echo Using Python 3.11:
echo   %BASEPY%
"%BASEPY%" -c "import sys; assert sys.version_info[:2]==(3,11); assert sys.maxsize > 2**32" >nul 2>nul
if errorlevel 1 goto :bad_python
"%BASEPY%" -m venv "runtime\venv"
if errorlevel 1 goto :failed

:have_venv
set "PY=%CD%\runtime\venv\Scripts\python.exe"
if not exist "%PY%" goto :failed

"%PY%" -m ensurepip --upgrade >nul 2>nul
if "%BGREMOVER_PIP_INSECURE%"=="1" goto :pip_bootstrap_compat
"%PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :failed
goto :detect_gpu

:pip_bootstrap_compat
"%PY%" -m pip install --upgrade pip setuptools wheel --trusted-host pypi.org --trusted-host files.pythonhosted.org
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
echo Installation complete.
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

:detect_python311
set "BASEPY="
del /q "%PY_PATH_FILE%" >nul 2>nul
where py >nul 2>nul
if errorlevel 1 goto :detect_plain_python
py -3.11 -c "import sys,pathlib; assert sys.version_info[:2]==(3,11); assert sys.maxsize > 2**32; pathlib.Path(r'%PY_PATH_FILE%').write_text(sys.executable, encoding='utf-8')" >nul 2>nul
if errorlevel 1 goto :detect_plain_python
set /p BASEPY=<"%PY_PATH_FILE%"
del /q "%PY_PATH_FILE%" >nul 2>nul
if defined BASEPY goto :eof

:detect_plain_python
where python >nul 2>nul
if errorlevel 1 goto :eof
python -c "import sys,pathlib; assert sys.version_info[:2]==(3,11); assert sys.maxsize > 2**32; pathlib.Path(r'%PY_PATH_FILE%').write_text(sys.executable, encoding='utf-8')" >nul 2>nul
if errorlevel 1 goto :eof
set /p BASEPY=<"%PY_PATH_FILE%"
del /q "%PY_PATH_FILE%" >nul 2>nul
goto :eof

:detect_known_python311_paths
set "BASEPY="
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "BASEPY=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if defined BASEPY goto :eof
if exist "%ProgramFiles%\Python311\python.exe" set "BASEPY=%ProgramFiles%\Python311\python.exe"
if defined BASEPY goto :eof
if exist "%ProgramFiles%\Python311-64\python.exe" set "BASEPY=%ProgramFiles%\Python311-64\python.exe"
goto :eof

:bad_python
echo Selected Python is not a 64-bit Python 3.11 installation.
goto :failed

:cancelled
echo.
echo Installation cancelled.
pause
exit /b 1

:failed
del /q "%PY_PATH_FILE%" >nul 2>nul
echo.
echo Installation failed.
echo If NVIDIA is installed, the lines above must show CUDA wheel and CUDA available: True.
pause
exit /b 1
