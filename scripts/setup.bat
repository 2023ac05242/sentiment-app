@echo off
setlocal ENABLEDELAYEDEXECUTION
cd /d "%~dp0\.."

REM Usage:
REM   scripts\setup.bat           -> auto: use model if present; download if missing/pointer
REM   scripts\setup.bat 1         -> force install HF deps + one-time model download
REM   scripts\setup.bat 0         -> VADER only (skip HF deps/model)

set "WITH_HF=%~1"
if "%WITH_HF%"=="" set "WITH_HF=auto"

echo [0/6] (Optional) Try to fetch any Git LFS files...
git lfs install >nul 2>&1
git lfs pull    >nul 2>&1

echo [1/6] Ensuring Python launcher is available...
where py >nul 2>nul || (
  echo   Python launcher 'py' not found. Please install Python 3.10+ (winget install -e --id Python.Python.3.10)
  exit /b 1
)

echo [2/6] Creating virtual environment (.venv)...
py -3 -m venv .venv || (
  echo   Failed to create venv.
  exit /b 1
)

echo [3/6] Activating venv and installing base deps...
call ".\.venv\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install -r backend\requirements.txt || exit /b 1

REM Decide if we need the HF model
set "MODEL=models\sst2\model.safetensors"
set "NEED_MODEL=0"
if /I "%WITH_HF%"=="1" (
  set "NEED_MODEL=1"
) else if /I "%WITH_HF%"=="auto" (
  if not exist "%MODEL%" (
    set "NEED_MODEL=1"
  ) else (
    for %%F in ("%MODEL%") do set "SIZE=%%~zF"
    findstr /m /c:"git-lfs" "%MODEL%" >nul 2>&1 && set "NEED_MODEL=1"
    if not defined SIZE set "NEED_MODEL=1"
    if defined SIZE if %SIZE% LSS 100000 set "NEED_MODEL=1"
  )
)

if "%NEED_MODEL%"=="1" (
  echo [4/6] Installing HF/torch deps and downloading model (one time)...
  pip install -r backend\requirements-hf.txt || exit /b 1
  pip install huggingface_hub==0.23.5 >nul 2>&1

  REM Make sure we’re allowed to go online for the download (in case user set offline vars globally)
  set HF_HUB_OFFLINE=
  set TRANSFORMERS_OFFLINE=

  python -c "from huggingface_hub import snapshot_download; snapshot_download('distilbert/distilbert-base-uncased-finetuned-sst-2-english', local_dir='models/sst2'); print('Model ready at models/sst2')" || (
    echo   Download failed. You can re-run this script later with "1" to try again.
  )
) else (
  echo [4/6] HF model step skipped (VADER only or model already present).
)

echo [5/6] (Optional) Install aiofiles for static frontend serving...
pip install aiofiles >nul 2>&1

echo [6/6] Starting the server at http://127.0.0.1:8000/ ...
start "" http://127.0.0.1:8000/
python backend\app.py
