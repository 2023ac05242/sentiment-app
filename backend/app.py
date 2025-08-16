"""
FastAPI Sentiment Analysis Backend (+ static frontend, offline HF option)
------------------------------------------------------------------------
Two engines:
1) VADER (default) – fast, small
2) Transformer (optional, local-only) – loads from a local model folder, no downloads

Endpoints:
- GET  /health
- POST /analyze  (form: text?, file?, column?, model in {"vader","hf"})

Static frontend is served from ../frontend at "/".
"""

import io
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Local-only Hugging Face settings ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Where your one-time download lives (see the commands I gave you)
HF_MODEL_DIR = Path(os.getenv("HF_MODEL_DIR", str(PROJECT_ROOT / "models" / "sst2")))

# Force offline: no network calls, ever
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Skip TF/Flax backends to avoid heavy deps
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

# Quiet Windows symlink warning (optional)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Caches (kept local; harmless when using local model dir)
HF_CACHE_DIR = os.environ.get("HF_HOME") or str(PROJECT_ROOT / "hf-cache")
os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
# (Removed deprecated TRANSFORMERS_CACHE to silence FutureWarning)
os.environ.setdefault("HF_DATASETS_CACHE", HF_CACHE_DIR)

# Startup preload to remove first-request lag (set False to disable)
PRELOAD_HF = True
# ─────────────────────────────────────────────────────────────────────────────

# NLTK / preprocessing
import nltk
from nltk.corpus import stopwords, wordnet  # noqa: F401
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

import pandas as pd

_USE_HF_DEFAULT = False  # keep default = VADER unless user chooses "hf" in UI
_HF_PIPELINE = None

app = FastAPI(title="Sentiment Analysis API", version="1.0")


def _safe_nltk_download(resource: str) -> None:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split("/")[-1])


# Ensure required NLTK resources exist
_safe_nltk_download("tokenizers/punkt")
_safe_nltk_download("tokenizers/punkt_tab")  # NLTK 3.9+
_safe_nltk_download("corpora/stopwords")
_safe_nltk_download("corpora/wordnet")
_safe_nltk_download("corpora/omw-1.4")
_safe_nltk_download("sentiment/vader_lexicon")


class Preprocessor:
    def __init__(self):
        self.stop = set(stopwords.words("english"))
        self.wnl = WordNetLemmatizer()
        self.url_pat = re.compile(r"https?://\S+|www\.\S+")
        self.mention_pat = re.compile(r"@[A-Za-z0-9_]+")
        self.nonalpha_pat = re.compile(r"[^a-zA-Z\s]")

    def __call__(self, text: str) -> Dict[str, Any]:
        original = text
        text = text.strip()
        text = self.url_pat.sub(" ", text)
        text = self.mention_pat.sub(" ", text)
        text = self.nonalpha_pat.sub(" ", text)
        text = re.sub(r"\s+", " ", text).lower()
        tokens = word_tokenize(text, preserve_line=True)  # no runtime punkt_tab lookup
        tokens = [t for t in tokens if t not in self.stop and len(t) > 1]
        lemmas = [self.wnl.lemmatize(t) for t in tokens]
        return {
            "original": original,
            "cleaned": text,
            "tokens": tokens,
            "lemmas": lemmas,
            "preprocessed_text": " ".join(lemmas),
        }


class VaderEngine:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def predict(self, text: str) -> Dict[str, Any]:
        s = self.analyzer.polarity_scores(text)
        c = s.get("compound", 0.0)
        label = "positive" if c >= 0.05 else "negative" if c <= -0.05 else "neutral"
        return {
            "engine": "vader",
            "scores": {"positive": s["pos"], "neutral": s["neu"], "negative": s["neg"], "compound": c},
            "label": label,
        }


class HFPipelineEngine:
    """
    Loads a LOCAL transformer model (no downloads).
    If HF_MODEL_DIR doesn't exist or loading fails => raise so caller can fallback.
    """
    def __init__(self):
        global _HF_PIPELINE
        if not HF_MODEL_DIR.is_dir():
            raise RuntimeError(f"Local HF model dir not found: {HF_MODEL_DIR}")

        if _HF_PIPELINE is None:
            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification,
                pipeline,
            )
            # Load from disk only (no network), then create pipeline WITHOUT extra kwargs
            tok = AutoTokenizer.from_pretrained(
                str(HF_MODEL_DIR), local_files_only=True
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                str(HF_MODEL_DIR), local_files_only=True
            )
            _HF_PIPELINE = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tok,
            )
        self.pipe = _HF_PIPELINE

    def predict(self, text: str) -> Dict[str, Any]:
        out = self.pipe(text)[0]  # {'label': 'POSITIVE', 'score': 0.99}
        lbl = out.get("label", "POSITIVE").lower()
        score = float(out.get("score", 0.5))
        if 0.45 <= score <= 0.55:
            return {
                "engine": "hf",
                "scores": {"positive": 0.0, "neutral": 1.0, "negative": 0.0, "confidence": score},
                "label": "neutral",
            }
        if lbl.startswith("pos"):
            pos, neg = score, 1 - score
            label = "positive"
        else:
            neg, pos = score, 1 - score
            label = "negative"
        neu = max(0.0, 1.0 - abs(score - 0.5) * 2)
        return {
            "engine": "hf",
            "scores": {"positive": pos, "neutral": neu, "negative": neg, "confidence": score},
            "label": label,
        }


def get_engine(name: Optional[str]) -> Any:
    name = (
        (name or os.getenv("SENTIMENT_ENGINE"))
        or ("hf" if os.getenv("USE_HF", "0") == "1" else None)
        or ("hf" if _USE_HF_DEFAULT else "vader")
    )
    name = (name or "vader").lower()
    if name == "hf":
        try:
            return HFPipelineEngine()
        except Exception as e:
            # No local model available or HF not installed → fall back to VADER
            print("[hf] disabled or unavailable -> falling back to VADER:", e)
            return VaderEngine()
    return VaderEngine()


class ItemResult(BaseModel):
    text: str
    preprocessed_text: str
    label: str
    scores: Dict[str, float]
    engine: str


def aggregate(results: List[ItemResult]) -> Dict[str, Any]:
    if not results:
        return {"positive": 0, "neutral": 0, "negative": 0, "label": "neutral"}
    pos = sum(r.scores.get("positive", 0.0) for r in results) / len(results)
    neu = sum(r.scores.get("neutral", 0.0) for r in results) / len(results)
    neg = sum(r.scores.get("negative", 0.0) for r in results) / len(results)
    label = max([("positive", pos), ("neutral", neu), ("negative", neg)], key=lambda x: x[1])[0]
    return {"positive": pos, "neutral": neu, "negative": neg, "label": label}


preproc = Preprocessor()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "default_engine": os.getenv("SENTIMENT_ENGINE", "vader"),
        "hf_local_dir": str(HF_MODEL_DIR),
        "hf_available": HF_MODEL_DIR.is_dir(),
    }


@app.post("/analyze")
async def analyze(
    text: Optional[str] = Form(default=None),
    model: Optional[str] = Form(default=None),
    column: Optional[str] = Form(default=None),
    file: Optional[UploadFile] = File(default=None),
):
    engine = get_engine(model)

    items: List[str] = []
    if text and text.strip():
        items = [text.strip()]

    if file is not None:
        if file.content_type in ("text/plain", "application/octet-stream") or (file.filename or "").lower().endswith(".txt"):
            data = (await file.read()).decode("utf-8", errors="ignore")
            for line in data.splitlines():
                if line.strip():
                    items.append(line.strip())
        elif file.content_type in ("text/csv", "application/vnd.ms-excel") or (file.filename or "").lower().endswith(".csv"):
            raw = await file.read()
            df = pd.read_csv(io.BytesIO(raw))
            colname = column
            if not colname:
                object_cols = [c for c in df.columns if df[c].dtype == object]
                if not object_cols:
                    raise HTTPException(status_code=400, detail="No text-like column found; please specify 'column'.")
                colname = max(object_cols, key=lambda c: df[c].astype(str).str.len().mean())
            if colname not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{colname}' not in CSV.")
            series = df[colname].astype(str).fillna("")
            for s in series:
                s = s.strip()
                if s:
                    items.append(s)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Upload .txt or .csv")

    if not items:
        raise HTTPException(status_code=400, detail="Provide 'text' or upload a .txt/.csv 'file'.")

    preds: List[ItemResult] = []
    for item in items:
        pp = preproc(item)
        p = engine.predict(pp["original"])  # keep punctuation/emojis for VADER
        preds.append(ItemResult(text=pp["original"], preprocessed_text=pp["preprocessed_text"], label=p["label"], scores=p["scores"], engine=p["engine"]))

    agg = aggregate(preds)
    return {"count": len(preds), "engine": preds[0].engine if preds else engine.__class__.__name__, "predictions": [r.model_dump() for r in preds], "aggregate": agg}


# Preload HF at startup (only if the local dir exists)
@app.on_event("startup")
def _preload_hf_on_startup():
    if PRELOAD_HF and HF_MODEL_DIR.is_dir():
        try:
            _ = HFPipelineEngine()
            print(f"[startup] HF pipeline preloaded from local dir: {HF_MODEL_DIR}")
        except Exception as e:
            print("[startup] HF preload skipped:", e)


# Serve frontend
FRONTEND_DIR = PROJECT_ROOT / "frontend"
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
