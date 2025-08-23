# Sentiment App — Quick Start (Windows / Anaconda Prompt)

This README gives you copy‑paste commands to get the app running on a Windows machine using **Anaconda Prompt**. It uses a Python virtual environment inside the project and can optionally download the Transformer model once.

> If you only need the fast/lightweight version, you can stop after installing `backend/requirements.txt` and run the app.

---

## 1) Clone the repository

**In Anaconda Prompt:**

```bat
cd C:\Users\91948\Downloads   # Use your own folder and path
```

```bat
git lfs install && git clone https://github.com/2023ac05242/sentiment-app.git && cd sentiment-app && git lfs pull
```

> If the folder already exists, either delete it first or just `cd` into it directly.

---

## 2) Create a virtual environment and install base dependencies

```bat
cd C:\Users\91948\Downloads\sentiment-app   # Use your own folder and path
```

```bat
python -m venv .venv
```

```bat
.\.venv\Scripts\activate
```

```bat
python -m pip install --upgrade pip
```

```bat
python -m pip install -r backend\requirements.txt
```

---

## 3) Optional: Transformer model (one‑time download)

> This step installs the Transformer engine and downloads the SST‑2 model locally. **It can take up to \~5 minutes** the first time (depends on network).

```bat
python -m pip install -r backend\requirements-hf.txt
```

```bat
python -c "from huggingface_hub import snapshot_download; snapshot_download('distilbert/distilbert-base-uncased-finetuned-sst-2-english', local_dir='models/sst2', local_dir_use_symlinks=False); print('Model downloaded to models/sst2')"
```

---

## 4) Run the app

```bat
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

Open in your browser (wait for for a minute after previous step for application to start):

```
http://127.0.0.1:8000/
```

The frontend (`frontend/index.html`) makes requests to `http://localhost:8000/analyze`. Keep the backend running while you use the UI.

---

## 5) Useful tips

* **Deactivate the venv:**

  ```bat
  deactivate
  ```
* **Re‑activate later:**

  ```bat
  .\.venv\Scripts\activate
  ```
* **Health check:**

  ```bat
  curl http://127.0.0.1:8000/health
  ```
* **If port 8000 is busy:**

  ```bat
  for /f "tokens=5" %p in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do taskkill /PID %p /F
  ```

---

## 6) Troubleshooting

* If `python` isn’t found: install Python 3.10+ or ensure Anaconda provides `python` in your PATH.
* If `git lfs pull` shows pointers instead of the model, ensure Git LFS is installed (`git lfs install`) and run it again.
* If you skip the Transformer step, the app will use **VADER** by default.
