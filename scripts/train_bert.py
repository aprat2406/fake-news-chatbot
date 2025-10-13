
# scripts/train_bert.py
"""
Fine-tune BERT (huggingface Trainer) on processed dataset.
Saves best model to models/bert-finetuned
"""
import os
import json
from pathlib import Path
import logging
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    AutoTokenizer,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
LOG = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "processed" / "bert_dataset"
LABEL_MAP_PATH = PROJECT_ROOT / "processed" / "label_map.json"
MODEL_NAME = "bert-base-uncased"
OUT_DIR = PROJECT_ROOT / "models" / "bert-finetuned"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def is_cuda_available():
    return torch.cuda.is_available()

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

def main():
    if not PROCESSED_DIR.exists():
        LOG.error(f"Processed dataset not found at {PROCESSED_DIR}. Run preprocess.py first.")
        return

    ds = load_from_disk(str(PROCESSED_DIR.parent))  # root folder processed/ containing bert_dataset
    LOG.info("Datasets loaded.")

    # load label map
    label_map = json.load(open(LABEL_MAP_PATH))
    num_labels = len(label_map)
    LOG.info(f"Number of labels: {num_labels}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # TrainingArguments - tweak these if OOM or slow
    training_args = TrainingArguments(
        output_dir=str(OUT_DIR),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=16,   # reduce to 8 or 4 if OOM
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        num_train_epochs=4,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=is_cuda_available(),  # use mixed precision if GPU available
        logging_steps=100,
        save_total_limit=3,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    LOG.info("Starting training...")
    trainer.train()
    LOG.info("Training finished. Saving model...")
    trainer.save_model(str(OUT_DIR))
    LOG.info(f"Model saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
