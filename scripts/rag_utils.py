"""
RAG Utilities for Fake News Detection Chatbot
---------------------------------------------
- Builds a FAISS index from your dataset (combined_true_fake_full.csv)
- Saves:
    vectors.index : FAISS vector file
    meta.json     : metadata for each vector (text, label, source)
Usage:
    python3 scripts/rag_utils.py build
    python3 scripts/rag_utils.py query "The government announced..."
"""

import os
import json
import sys
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "combined_true_fake_full.csv"
INDEX_PATH = PROJECT_ROOT / "vectors.index"
META_PATH = PROJECT_ROOT / "meta.json"

# Embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def build_index(sample_size=None):
    """
    Builds FAISS index from dataset.
    sample_size: int (optional) - to limit number of rows for testing
    """
    if not DATA_PATH.exists():
        print(f"[ERROR] Dataset not found at {DATA_PATH}")
        sys.exit(1)

    print(f"🔍 Reading dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if sample_size:
        df = df.sample(min(sample_size, len(df)), random_state=42)
    df = df.dropna(subset=["text"]).reset_index(drop=True)

    print(f"✅ Loaded {len(df)} rows.")

    # Load embedder
    print(f"🧠 Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Compute embeddings
    print("🔢 Encoding text to embeddings...")
    embeddings = model.encode(df["text"].tolist(), show_progress_bar=True, convert_to_numpy=True, batch_size=64)
    dim = embeddings.shape[1]

    # Build FAISS index
    print(f"🗂️ Building FAISS index with {len(embeddings)} vectors (dim={dim})")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save index + metadata
    print(f"💾 Saving index to {INDEX_PATH}")
    faiss.write_index(index, str(INDEX_PATH))

    print(f"💾 Saving metadata to {META_PATH}")
    meta = []
    for i, row in df.iterrows():
        meta.append({
            "id": i,
            "text": str(row["text"])[:600],
            "label": str(row.get("label", "")),
            "source": str(row.get("source_file", "")),
        })
    with open(META_PATH, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print("✅ RAG index build complete.")
    print(f"Total entries: {len(meta)}")

def query_index(query, top_k=3):
    """
    Query FAISS index and return top-k matches
    """
    if not INDEX_PATH.exists() or not META_PATH.exists():
        print("❌ FAISS index or meta.json not found. Please run: python3 scripts/rag_utils.py build")
        return []

    print("🧠 Loading model and FAISS index...")
    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(str(INDEX_PATH))
    meta = json.load(open(META_PATH, "r"))

    print(f"🔎 Querying: {query}")
    q_vec = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_vec, top_k)

    results = []
    for score, idx in zip(D[0], I[0]):
        if 0 <= idx < len(meta):
            entry = meta[idx]
            results.append({
                "score": float(score),
                "text": entry["text"],
                "label": entry.get("label", ""),
                "source": entry.get("source", "")
            })

    for i, r in enumerate(results, start=1):
        print(f"\n{i}. {r['text'][:250]}...")
        print(f"   Label: {r['label']} | Source: {r['source']} | Score: {round(r['score'],3)}")

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:\n  python3 scripts/rag_utils.py build\n  python3 scripts/rag_utils.py query 'some text'")
        sys.exit(0)

    cmd = sys.argv[1].lower()
    if cmd == "build":
        sample = int(sys.argv[2]) if len(sys.argv) > 2 else None
        build_index(sample_size=sample)
    elif cmd == "query":
        query = sys.argv[2] if len(sys.argv) > 2 else input("Enter query: ")
        query_index(query)
    else:
        print(f"Unknown command: {cmd}")
