"""
Fine-tune Amazon Chronos-2 on Store Sales (Favorita) data.

Strategy:
  1. Load base checkpoint: amazon/chronos-t5-small
  2. Tokenize our time series using ChronosTokenizer (quantile-bin encoding)
  3. Fine-tune the T5 model with Seq2SeqTrainer (HuggingFace)
  4. Evaluate fine-tuned vs zero-shot on test set

Why fine-tune?
  Zero-shot Chronos-2 RMSLE = 0.2040 — it never saw Ecuadorian grocery data.
  Fine-tuning adapts the model's token distributions to our specific
  demand patterns (oil-price sensitivity, local holidays, promotion spikes).

Usage:
    python -m src.finetune_chronos

Reference:
    https://github.com/amazon-science/chronos-forecasting
"""
from __future__ import annotations

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import mlflow
from pathlib import Path
from torch.utils.data import Dataset

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    EarlyStoppingCallback,
)
from chronos import ChronosPipeline, ChronosTokenizer

from src.config import (
    MODELS_DIR, MLFLOW_EXPERIMENT,
    ID_COL, DATE_COL, TARGET_COL,
    HORIZON, RANDOM_SEED,
)
from src.data_loader import load_data
from src.metrics import evaluate_forecasts, print_metrics_table

# ── Config ─────────────────────────────────────────────────────────────────

BASE_MODEL      = "amazon/chronos-t5-small"
FINETUNED_PATH  = MODELS_DIR / "chronos_finetuned"
CONTEXT_LENGTH  = 512   # tokens of history fed to the model
NUM_SAMPLES     = 50    # samples for probabilistic forecast
BATCH_SIZE      = 16
LEARNING_RATE   = 1e-4
NUM_EPOCHS      = 3     # 3 epochs sufficient for fine-tuning on ~300 series
WARMUP_STEPS    = 50
MAX_STEPS       = 1000  # cap to avoid very long training


# ── Dataset ─────────────────────────────────────────────────────────────────

class ChronosFineTuneDataset(Dataset):
    """
    Sliding-window dataset for Chronos fine-tuning.

    Each sample = (context_ids, target_ids) where:
      - context_ids: tokenized past context_length observations
      - target_ids:  tokenized next train_horizon observations

    IMPORTANT: train_horizon must match tokenizer.config.prediction_length (64 for
    chronos-t5-small). We use 64-step targets for training, then trim to 28 at eval.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: ChronosTokenizer,
        context_length: int = CONTEXT_LENGTH,
        horizon: int = HORIZON,
    ):
        self.samples   = []
        self.tokenizer = tokenizer
        # Use model's native prediction_length for training targets
        train_horizon  = tokenizer.config.prediction_length  # 64 for chronos-t5-small

        for uid, grp in df.groupby(ID_COL, observed=True):
            values = grp.sort_values(DATE_COL)[TARGET_COL].values.astype(np.float32)
            n = len(values)

            # Sliding windows — step = train_horizon to avoid excessive overlap
            step = train_horizon
            for start in range(0, n - context_length - train_horizon + 1, step):
                ctx_slice = values[start: start + context_length]
                tgt_slice = values[start + context_length:
                                   start + context_length + train_horizon]
                self.samples.append((ctx_slice, tgt_slice))

        print(f"  Fine-tune dataset: {len(self.samples):,} windows "
              f"from {df[ID_COL].nunique()} series")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ctx, tgt = self.samples[idx]

        ctx_tensor = torch.tensor(ctx, dtype=torch.float32)
        tgt_tensor = torch.tensor(tgt, dtype=torch.float32)

        # Tokenize context
        ctx_ids, ctx_mask, scale = self.tokenizer.context_input_transform(
            ctx_tensor.unsqueeze(0)
        )
        # Tokenize target using the same scale
        tgt_ids, tgt_mask = self.tokenizer.label_input_transform(
            tgt_tensor.unsqueeze(0), scale
        )

        return {
            "input_ids"      : ctx_ids.squeeze(0),
            "attention_mask" : ctx_mask.squeeze(0),
            "labels"         : tgt_ids.squeeze(0),
        }


# ── Fine-tune ────────────────────────────────────────────────────────────────

def finetune(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Path:
    """
    Fine-tune Chronos on the training set, validate on val_df.
    Returns path to saved fine-tuned model.
    """
    print(f"\nLoading base model: {BASE_MODEL}")
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = ChronosPipeline.from_pretrained(
        BASE_MODEL,
        device_map=device_str,
        dtype=torch.float32,   # float32 for stable gradient computation
    )
    model     = pipeline.model
    tokenizer = pipeline.tokenizer

    # Ensure all parameters require grad and model is in training mode
    model.train()
    for param in model.parameters():
        param.requires_grad_(True)

    device = next(model.parameters()).device
    print(f"  Model loaded on {device} (float32, {sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")

    # Build datasets
    print("\nBuilding fine-tune dataset...")
    train_ds = ChronosFineTuneDataset(train_df, tokenizer)
    # Use last 5% of train series as validation proxy
    val_ids  = train_df[ID_COL].unique()[:max(1, len(train_df[ID_COL].unique())//20)]
    val_ds   = ChronosFineTuneDataset(
        train_df[train_df[ID_COL].isin(val_ids)], tokenizer,
        context_length=CONTEXT_LENGTH, horizon=HORIZON,
    )

    # Training arguments
    FINETUNED_PATH.mkdir(parents=True, exist_ok=True)
    args = Seq2SeqTrainingArguments(
        output_dir                  = str(FINETUNED_PATH / "checkpoints"),
        num_train_epochs            = NUM_EPOCHS,
        max_steps                   = MAX_STEPS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        learning_rate               = LEARNING_RATE,
        warmup_steps                = WARMUP_STEPS,
        lr_scheduler_type           = "cosine",
        eval_strategy               = "steps",
        eval_steps                  = 100,
        save_strategy               = "steps",
        save_steps                  = 100,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        predict_with_generate       = False,
        fp16                        = False,
        bf16                        = torch.cuda.is_available(),
        report_to                   = "none",   # avoid wandb
        seed                        = RANDOM_SEED,
        logging_steps               = 50,
        dataloader_num_workers      = 0,        # avoid multiprocessing OOM
    )

    trainer = Seq2SeqTrainer(
        model           = model,
        args            = args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        data_collator   = default_data_collator,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print(f"\nFine-tuning ({len(train_ds):,} windows, {NUM_EPOCHS} epochs, "
          f"max {MAX_STEPS} steps)...")
    trainer.train()

    # Save fine-tuned model + tokenizer
    model.save_pretrained(str(FINETUNED_PATH))
    print(f"  Saved fine-tuned model -> {FINETUNED_PATH}")
    return FINETUNED_PATH


# ── Evaluate fine-tuned model ─────────────────────────────────────────────

def evaluate_finetuned(
    finetuned_path: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Load fine-tuned model and run forecast on test set."""
    print(f"\nLoading fine-tuned model from {finetuned_path}...")
    pipeline = ChronosPipeline.from_pretrained(
        str(finetuned_path),
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32,
    )

    print(f"Running forecast on {train_df[ID_COL].nunique()} series...")
    results = []
    for uid, grp in train_df.groupby(ID_COL, observed=True):
        grp = grp.sort_values(DATE_COL)
        values = grp[TARGET_COL].values[-CONTEXT_LENGTH:].astype(np.float32)
        context = torch.tensor(values, dtype=torch.float32).unsqueeze(0)

        samples = pipeline.predict(
            context,
            prediction_length=HORIZON,
            num_samples=NUM_SAMPLES,
        )[0].numpy()
        samples = np.clip(samples, 0, None)

        last_date = grp[DATE_COL].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1),
                                     periods=HORIZON, freq="D")
        for t, dt in enumerate(future_dates):
            results.append({
                ID_COL  : uid,
                DATE_COL: dt,
                "y_pred": float(np.median(samples[:, t])),
                "lo-90" : float(np.quantile(samples[:, t], 0.05)),
                "hi-90" : float(np.quantile(samples[:, t], 0.95)),
            })

    preds_df = pd.DataFrame(results)
    metrics  = evaluate_forecasts(test_df, preds_df, train_df)
    print_metrics_table(metrics, "Chronos-2 (fine-tuned)")

    preds_df.to_parquet(MODELS_DIR / "preds_chronos_finetuned.parquet", index=False)
    print(f"  Saved: {MODELS_DIR}/preds_chronos_finetuned.parquet")
    return metrics, preds_df


# ── Main ──────────────────────────────────────────────────────────────────

def run():
    train_df, test_df = load_data()

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="Chronos-finetuned"):
        mlflow.log_params({
            "model"          : BASE_MODEL,
            "context_length" : CONTEXT_LENGTH,
            "num_epochs"     : NUM_EPOCHS,
            "max_steps"      : MAX_STEPS,
            "learning_rate"  : LEARNING_RATE,
            "batch_size"     : BATCH_SIZE,
        })

        # Fine-tune
        ft_path = finetune(train_df, test_df)

        # Evaluate
        metrics, _ = evaluate_finetuned(ft_path, train_df, test_df)
        agg = metrics[metrics[ID_COL] == "ALL (mean)"].iloc[0]
        mlflow.log_metrics({k: float(v) for k, v in agg.items()
                             if k != ID_COL and pd.notna(v)})

        # Compare with zero-shot
        print("\n=== CHRONOS COMPARISON ===")
        print(f"  Zero-shot  RMSLE=0.2040  MASE=1.038")
        print(f"  Fine-tuned RMSLE={agg['rmsle']:.4f}  MASE={agg['mase']:.4f}")
        delta = 0.2040 - float(agg["rmsle"])
        print(f"  Improvement: {delta:+.4f} RMSLE ({delta/0.2040*100:+.1f}%)")

    return metrics


if __name__ == "__main__":
    run()
