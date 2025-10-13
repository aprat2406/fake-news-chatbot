# scripts/preprocess.py
"""
Preprocessing for fake-news dataset.
- Reads CSV(s) from ../data
- Cleans text
- Maps labels to ints
- Tokenizes with a HuggingFace tokenizer (bert-base-uncased)
- Splits into train/validation/test (70/15/15)
- Saves processed dataset to ../processed/bert_dataset and label_map.json
"""
import re
import json
import os
from pathlib import Path
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
LOG = logging.getLogger(__name__)

# Config
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_NAME = "bert-base-uncased"
SEED = 42
MAX_LEN_SHORT = 128
MAX_LEN_LONG = 512  # for later use with long models

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)      # remove HTML
    text = re.sub(r"http\S+", " ", text)      # remove URLs
    text = re.sub(r"[^A-Za-z0-9,.!?;:'\"()\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

def collect_csvs(data_dir: Path):
    csvs = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.tsv"))
    LOG.info(f"Found {len(csvs)} csv/tsv files in {data_dir}")
    return csvs

def read_file(path: Path):
    try:
        if path.suffix.lower() == ".tsv":
            df = pd.read_csv(path, sep="\t", low_memory=False)
        else:
            df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        LOG.error(f"Failed to read {path}: {e}")
        return pd.DataFrame()
    return df

def pick_text_column(df: pd.DataFrame):
    for c in df.columns:
        if c.lower() in ("text", "content", "article", "body", "headline", "title"):
            return c
    return df.columns[0] if len(df.columns) > 0 else None

def normalize_label(raw_label):
    if pd.isna(raw_label):
        return None
    s = str(raw_label).strip().lower()
    # common conversions - add more mappings as required
    if s in ("true", "real", "supported"):
        return "true"
    if s in ("false", "fake", "refuted", "pants-fire"):
        return "fake"
    if s in ("not enough info", "not_enough_info", "not_enough_information", "not_enough"):
        return "not_enough_info"
    # keep original if we don't recognize - user can map later
    return s

def build_combined_dataframe():
    csv_paths = collect_csvs(DATA_DIR)
    all_rows = []
    for p in csv_paths:
        df = read_file(p)
        if df.empty:
            continue
        text_col = pick_text_column(df)
        label_col = None
        for c in df.columns:
            if c.lower() == "label":
                label_col = c
                break
        if text_col is None:
            continue
        LOG.info(f"Reading {p.name} - text_col={text_col} label_col={label_col}")
        for _, row in df.iterrows():
            text = row.get(text_col, "")
            if pd.isna(text) or str(text).strip()=="":
                continue
            label = None
            if label_col:
                label = normalize_label(row.get(label_col))
            # If no label column, try to infer from filename
            if not label:
                fname = p.name.lower()
                if fname.startswith("fake"):
                    label = "fake"
                elif fname.startswith("true"):
                    label = "true"
            all_rows.append({"text": clean_text(text), "label": label, "source_file": p.name})
    combined = pd.DataFrame(all_rows).drop_duplicates(subset=["text"]).reset_index(drop=True)
    LOG.info(f"Combined dataframe: {len(combined)} rows")
    return combined

def prepare_dataset(df: pd.DataFrame, output_dir: Path):
    # Keep only rows with a label for supervised training
    labelled = df.dropna(subset=["label"]).copy()
    LOG.info(f"Labelled rows: {len(labelled)}")
    # build label map
    labels = sorted(list(labelled['label'].unique()))
    label_map = {lab: i for i, lab in enumerate(labels)}
    LOG.info(f"Label map: {label_map}")
    labelled['label_id'] = labelled['label'].map(label_map)
    ds = Dataset.from_pandas(labelled[['text', 'label_id']])
    ds = ds.train_test_split(test_size=0.30, seed=SEED)
    test_valid = ds['test'].train_test_split(test_size=0.5, seed=SEED)
    data = DatasetDict({
        'train': ds['train'],
        'validation': test_valid['train'],
        'test': test_valid['test'],
    })
    LOG.info("Tokenizing (this may take a while)...")
    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN_SHORT)
    data = data.map(tokenize_fn, batched=True, remove_columns=["text"])
    data = data.rename_column("label_id", "labels")
    data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    # save
    save_path = output_dir / "bert_dataset"
    data.save_to_disk(str(save_path))
    with open(output_dir / "label_map.json", "w") as fh:
        json.dump(label_map, fh, indent=2)
    LOG.info(f"Saved processed dataset to: {save_path}")
    return data

if __name__ == "__main__":
    LOG.info("Starting preprocessing...")
    combined_df = build_combined_dataframe()
    if combined_df.empty:
        LOG.error("No data found in /data. Please place your CSV/TSV files there.")
        raise SystemExit(1)
    prepare_dataset(combined_df, OUTPUT_DIR)
    LOG.info("Preprocessing finished.")

