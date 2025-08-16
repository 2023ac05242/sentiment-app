@echo off
setlocal ENABLEDELAYEDEXPANSION

REM --- Config flags (0/1) ---
set WITH_HF=%1
if "%WITH_HF%"=="" set WITH_HF=0
set DOWNLOAD_MODEL=%2
if "%DOWNLOAD_MODEL%"=="" set DOWNLOAD_MODEL=0

echo [1/5] Ensuring Python 3 is available...
where py >nul 2>nul || (
  echo Python launcher not found. Installing Python 3.10 via winget...
  winget install -e --id Python.Python.3.10 || (
    echo Failed to install Python automatically. Please install Python 3.10+ and rerun.
    exit /b 1
  )
)
for /f "tokens=2 delims= " %%v in ('py -V') do set PYVER=%%v
echo Using Python %%PYVER%%

echo [2/5] Creating virtual environment (.venv)...
py -3 -m venv .venv || (
  echo Failed to create venv.
  exit /b 1
)

echo [3/5] Upgrading pip & installing base deps...
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r backend\requirements.txt || exit /b 1

if "%WITH_HF%"=="1" (
  echo [4/5] Installing optional Transformer deps...
  .\.venv\Scripts\python -m pip install -r backend\requirements-hf.txt || exit /b 1

  if "%DOWNLOAD_MODEL%"=="1" (
    echo [4b/5] One-time local model download (no re-download later)...
    .\.venv\Scripts\python - << PY
from huggingface_hub import snapshot_download
snapshot_download("distilbert/distilbert-base-uncased-finetuned-sst-2-english", local_dir="models/sst2")
print("Model downloaded to models/sst2")
PY
  )
)

echo [5/5] Starting the server...
.\.venv\Scripts\python backend\app.py
