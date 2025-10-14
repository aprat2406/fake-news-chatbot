# scripts/evaluate.py
"""
Evaluate the fine-tuned BERT model on the saved test split.
Outputs:
 - results/eval_metrics.txt       (text summary)
 - results/predictions.csv        (each test row with true/pred/p_confidence)
 - results/confusion_matrix.png   (confusion matrix image)
Usage:
  python3 scripts/evaluate.py --batch_size 32
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# Config / paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "processed" / "bert_dataset"
LABEL_MAP_PATH = PROJECT_ROOT / "processed" / "label_map.json"
MODEL_DIR = PROJECT_ROOT / "models" / "bert-finetuned"
OUT_DIR = PROJECT_ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_tokenizer(model_dir):
    if not Path(model_dir).exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

def prepare_dataloader(ds_split, batch_size):
    # ds_split expected to have fields: input_ids, attention_mask, labels
    input_ids = [np.array(x, dtype=np.int64) for x in ds_split["input_ids"]]
    attn = [np.array(x, dtype=np.int64) for x in ds_split["attention_mask"]]
    labels = np.array(ds_split["labels"], dtype=np.int64)

    # pad into tensors (they should already be fixed-length from preprocessing)
    input_ids = torch.tensor(np.stack(input_ids), dtype=torch.long)
    attn = torch.tensor(np.stack(attn), dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(input_ids, attn, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader

def inference_loop(model, loader):
    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            input_ids, attn, labels = [t.to(DEVICE) for t in batch]
            outputs = model(input_ids=input_ids, attention_mask=attn)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def save_predictions(outfile_csv, ds_split, preds, probs, inv_label_map):
    # Try to include original text if available
    texts = []
    if "text" in ds_split.column_names:
        texts = ds_split["text"]
    else:
        # no text column stored; put empty strings
        texts = [""] * len(preds)

    rows = []
    for i, (t, p, pr) in enumerate(zip(texts, preds, probs)):
        pred_label = inv_label_map.get(int(p), str(p))
        # keep confidence = max prob
        confidence = float(max(pr))
        rows.append({"index": i, "text": t, "pred_label": pred_label, "confidence": confidence})
    df = pd.DataFrame(rows)
    df.to_csv(outfile_csv, index=False)

def plot_confusion_matrix(cm, labels, outpath):
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, interpolation='nearest', aspect='auto')
    ax.set_title("Confusion Matrix")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    # annotate cells
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fig.tight_layout()
    plt.savefig(outpath, bbox_inches='tight')
    plt.close(fig)

def main(args):
    # Load label_map
    if not LABEL_MAP_PATH.exists():
        raise FileNotFoundError("processed/label_map.json not found. Run preprocess.py first.")
    label_map = json.load(open(LABEL_MAP_PATH))
    inv_label_map = {int(v): k for k, v in label_map.items()}
    sorted_label_names = [inv_label_map[i] for i in sorted(inv_label_map.keys())]

    # Load model + tokenizer
    print("Loading model...")
    tokenizer, model = load_model_and_tokenizer(MODEL_DIR)

    # Load processed dataset
    if not PROCESSED_DIR.exists():
        raise FileNotFoundError("Processed dataset not found at processed/bert_dataset. Run preprocess.py first.")
    ds = load_from_disk(str(PROCESSED_DIR))
    if "test" not in ds:
        raise ValueError("No 'test' split found in processed dataset.")
    test_split = ds["test"]
    print(f"Test split size: {len(test_split)}")

    # Prepare dataloader
    loader = prepare_dataloader(test_split, args.batch_size)

    # Run inference
    print("Running inference on test set...")
    true_labels, preds, probs = inference_loop(model, loader)

    # Metrics
    acc = accuracy_score(true_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average='weighted', zero_division=0)
    cls_report = classification_report(true_labels, preds, target_names=sorted_label_names, zero_division=0)
    cm = confusion_matrix(true_labels, preds)

    # Save metrics to text file
    metrics_txt = OUT_DIR / "eval_metrics.txt"
    with open(metrics_txt, "w") as fh:
        fh.write(f"Accuracy: {acc:.4f}\n")
        fh.write(f"Precision (weighted): {precision:.4f}\n")
        fh.write(f"Recall (weighted): {recall:.4f}\n")
        fh.write(f"F1-score (weighted): {f1:.4f}\n\n")
        fh.write("Classification Report:\n")
        fh.write(cls_report)
    print("Metrics saved to:", metrics_txt)

    # Save confusion matrix plot
    cm_path = OUT_DIR / "confusion_matrix.png"
    plot_confusion_matrix(cm, sorted_label_names, cm_path)
    print("Confusion matrix saved to:", cm_path)

    # Save predictions CSV (text may be empty if not present)
    preds_csv = OUT_DIR / "predictions.csv"
    save_predictions(preds_csv, test_split, preds, probs, inv_label_map)
    print("Predictions saved to:", preds_csv)

    # Print short summary to console
    print("\nSUMMARY:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1-score (weighted): {f1:.4f}")
    print("\nClassification report:")
    print(cls_report)
    print("\nSaved artifacts in folder:", OUT_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    args = parser.parse_args()
    main(args)

