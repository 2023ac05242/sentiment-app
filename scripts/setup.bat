@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0\.."

REM Usage:
REM   scripts\setup.bat        -> auto: use HF if local model present; else VADER (no download)
REM   scripts\setup.bat 1      -> install HF deps + one-time model download
REM   scripts\setup.bat 0      -> VADER only (skip HF deps/model)

set "WITH_HF=%~1"
if "%WITH_HF%"=="" set "WITH_HF=auto"

echo [0/6] (Optional) Try to fetch any Git LFS files...
git lfs install >nul 2>&1
git lfs pull    >nul 2>&1

echo [1/6] Finding Python...
set "VPY="
REM Prefer Conda python if env is active
if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" set "VPY=%CONDA_PREFIX%\python.exe"
REM Otherwise try plain python on PATH
if not defined VPY for /f "delims=" %%I in ('where python 2^>nul') do ( set "VPY=%%I" & goto :gotpy )
REM Otherwise try py launcher
if not defined VPY for /f "delims=" %%I in ('where py 2^>nul') do ( set "VPY=py" & set "PYARGS=-3" & goto :gotpy )

:gotpy
if not defined VPY (
  echo   Python not found. Install Python 3.10+ first:
  echo   winget install -e --id Python.Python.3.10
  exit /b 1
)
if /I "%VPY%"=="py" (
  for /f "delims=" %%V in ('py -3 -c "import sys;print(sys.version.split()[0])"') do set "PYVER=%%V"
  echo   Using py -3 (v!PYVER!)
) else (
  for /f "delims=" %%V in ('"%VPY%" -c "import sys;print(sys.version.split()[0])"') do set "PYVER=%%V"
  echo   Using "%VPY%" (v!PYVER!)
)

echo [2/6] Creating virtual environment (.venv)...
if /I "%VPY%"=="py" (
  call py -3 -m venv .venv || ( echo   Failed to create venv. & exit /b 1 )
) else (
  call "%VPY%" -m venv .venv || ( echo   Failed to create venv. & exit /b 1 )
)

echo [3/6] Activating venv and installing base deps...
call ".\.venv\Scripts\activate.bat"
set "VPYV=.venv\Scripts\python.exe"
"%VPYV%" -m pip install --upgrade pip
"%VPYV%" -m pip install -r backend\requirements.txt || exit /b 1

REM Decide if we need HF model
set "MODEL=models\sst2\model.safetensors"
set "NEED_MODEL=0"
if /I "%WITH_HF%"=="1" (
  set "NEED_MODEL=1"
) else if /I "%WITH_HF%"=="auto" (
  if not exist "%MODEL%" (
    set "NEED_MODEL=1"
  ) else (
    for %%F in ("%MODEL%") do set "SIZE=%%~zF"
    REM Assume real model if >100MB
    if not defined SIZE set "NEED_MODEL=1"
    if defined SIZE if !SIZE! LSS 100000000 set "NEED_MODEL=1"
  )
)

if "%NEED_MODEL%"=="1" (
  echo [4/6] Installing HF/torch deps and downloading model (one time)...
  "%VPYV%" -m pip install -r backend\requirements-hf.txt || exit /b 1
  "%VPYV%" -m pip install huggingface_hub==0.23.5 >nul 2>&1

  REM Ensure online for initial download
  set HF_HUB_OFFLINE=
  set TRANSFORMERS_OFFLINE=

  "%VPYV%" -c "from huggingface_hub import snapshot_download; snapshot_download('distilbert/distilbert-base-uncased-finetuned-sst-2-english', local_dir='models/sst2'); print('Model ready at models/sst2')"
) else (
  echo [4/6] HF model step skipped (VADER only or model already present).
)

echo [5/6] (Optional) Install aiofiles for static frontend serving...
"%VPYV%" -m pip install aiofiles >nul 2>&1

echo [6/6] Starting the server at http://127.0.0.1:8000/ ...
start "" http://127.0.0.1:8000/
"%VPYV%" backend\app.py
