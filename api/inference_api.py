# api/inference_api.py
"""
FastAPI inference server for the Fake-News Detection Chatbot.

Features:
- Loads a fine-tuned Hugging Face model from models/bert-finetuned
- Returns label, confidence, probabilities
- Optional lightweight local RAG: if a FAISS index (vectors.index) and meta.json
  exist in the project root, it returns top-k evidence snippets.
- Simple /health endpoint.

Run:
    uvicorn api.inference_api:app --host 0.0.0.0 --port 8000
"""
import os
import json
import logging
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Optional imports for RAG (only used if index + meta exist)
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    _HAS_RAG = True
except Exception:
    _HAS_RAG = False

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
LOG = logging.getLogger("inference_api")

# Paths / config
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "bert-finetuned"
LABEL_MAP_PATH = PROJECT_ROOT / "processed" / "label_map.json"

# RAG files (optional)
FAISS_INDEX_PATH = PROJECT_ROOT / "vectors.index"     # optional: created by rag_utils.build_index()
RAG_META_PATH = PROJECT_ROOT / "meta.json"           # optional: list of docs with fields {id, text, source}

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG.info(f"Using device: {DEVICE}")

# Load model + tokenizer
if not MODEL_DIR.exists():
    LOG.error(f"Model directory not found: {MODEL_DIR}. Please train and save model first.")
    raise SystemExit(1)

LOG.info("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
model.to(DEVICE)
model.eval()
LOG.info("Model loaded.")

# Load label map
if not LABEL_MAP_PATH.exists():
    LOG.error(f"Label map not found at {LABEL_MAP_PATH}. Make sure preprocess.py produced label_map.json.")
    label_map = {}
else:
    label_map = json.load(open(LABEL_MAP_PATH, "r"))
# invert mapping {label_name: id} -> {id: label_name}
inv_label_map = {int(v): k for k, v in label_map.items()}

# Initialize optional RAG resources lazily
_rag_index = None
_rag_meta = None
_rag_embedder = None

def _init_rag():
    global _rag_index, _rag_meta, _rag_embedder
    if not _HAS_RAG:
        LOG.info("RAG dependencies not available (sentence-transformers/faiss). Skipping RAG init.")
        return
    if _rag_index is not None:
        return
    if FAISS_INDEX_PATH.exists() and RAG_META_PATH.exists():
        LOG.info("Loading FAISS index and meta for RAG...")
        _rag_index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(RAG_META_PATH, "r", encoding="utf-8") as fh:
            _rag_meta = json.load(fh)
        _rag_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        LOG.info("RAG index loaded.")
    else:
        LOG.info("No FAISS index or meta.json found for RAG. You can create one using scripts/rag_utils.py.")

def query_rag(query: str, top_k: int = 3):
    """
    Return list of evidence dicts: [{id, text, source, score}, ...]
    If RAG resources are not available, returns [].
    """
    if not _HAS_RAG:
        return []
    _init_rag()
    if _rag_index is None or _rag_meta is None or _rag_embedder is None:
        return []
    qv = _rag_embedder.encode([query], convert_to_numpy=True)
    D, I = _rag_index.search(qv, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(_rag_meta):
            continue
        doc = _rag_meta[idx]
        results.append({
            "id": doc.get("id", idx),
            "text": doc.get("text", "")[:600],   # short snippet
            "source": doc.get("source", ""),
            "score": float(score)
        })
    return results

# FastAPI app
app = FastAPI(title="Fake News Detection - Inference API")

class Query(BaseModel):
    text: str
    use_rag: Optional[bool] = False
    top_k: Optional[int] = 3
    model: Optional[str] = "bert"   # placeholder: supports 'bert' now; can extend to 'llama' if available

def predict_single(text: str):
    """
    Returns: (label_name, confidence, probs list)
    """
    # Tokenize
    enc = tokenizer.encode_plus(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)
        logits = out.logits.detach().cpu().numpy()[0]
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy().tolist()
        pred_id = int(np.argmax(probs))
        confidence = float(max(probs))
        label = inv_label_map.get(pred_id, str(pred_id))
    return label, confidence, probs

@app.post("/predict")
def predict(q: Query):
    text = q.text.strip()
    if not text:
        return {"error": "empty text"}
    # Currently only BERT model is supported here (fine-tuned version)
    label, confidence, probs = predict_single(text)

    evidence = []
    if q.use_rag:
        evidence = query_rag(text, top_k=q.top_k)

    return {
        "label": label,
        "confidence": confidence,
        "probs": probs,
        "evidence": evidence
    }

@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "model_loaded": True, "labels": inv_label_map}

# optional: a simple debug endpoint to show model info (not public)
@app.get("/info")
def info():
    return {
        "model_dir": str(MODEL_DIR),
        "label_map": label_map,
        "device": str(DEVICE),
        "rag_available": bool(_HAS_RAG and FAISS_INDEX_PATH.exists() and RAG_META_PATH.exists())
    }

