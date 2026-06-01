"""
Continue fine-tuning Chronos-2 from the saved checkpoint.

Extends training from 1000 steps to 3000 steps total.
Evaluates after each additional 1000 steps to find the optimal checkpoint.

Usage:
    python -m src.tune_chronos
"""
from __future__ import annotations

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import mlflow

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator
from chronos import ChronosPipeline

from src.config import (
    MODELS_DIR, MLFLOW_EXPERIMENT,
    ID_COL, DATE_COL, TARGET_COL, HORIZON, RANDOM_SEED,
)
from src.data_loader import load_data
from src.finetune_chronos import (
    ChronosFineTuneDataset,
    evaluate_finetuned,
    FINETUNED_PATH,
    CONTEXT_LENGTH,
    NUM_SAMPLES,
)
from src.metrics import evaluate_forecasts, print_metrics_table

EXTENDED_PATH   = MODELS_DIR / "chronos_extended"
ADDITIONAL_STEPS = 2000   # train 2000 more steps (total ~3000)
LEARNING_RATE   = 5e-5    # lower LR for continued fine-tuning
BATCH_SIZE      = 16


def run():
    train_df, test_df = load_data()

    print("=" * 55)
    print("  Chronos-2 Extended Fine-tuning")
    print(f"  Starting from: {FINETUNED_PATH}")
    print(f"  Additional steps: {ADDITIONAL_STEPS}")
    print(f"  Learning rate: {LEARNING_RATE} (reduced for continuation)")
    print("=" * 55)

    # Load from saved fine-tuned checkpoint
    print(f"\nLoading checkpoint from {FINETUNED_PATH}...")
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline  = ChronosPipeline.from_pretrained(
        str(FINETUNED_PATH),
        device_map=device_str,
        dtype=torch.float32,
    )
    model     = pipeline.model.model   # inner T5
    tokenizer = pipeline.tokenizer

    model.train()
    for param in model.parameters():
        param.requires_grad_(True)

    device = next(model.parameters()).device
    print(f"  Model loaded on {device}")

    # Dataset
    print("\nBuilding dataset...")
    train_ds = ChronosFineTuneDataset(train_df, tokenizer,
                                      context_length=CONTEXT_LENGTH)
    val_ids  = train_df[ID_COL].unique()[:max(1, len(train_df[ID_COL].unique())//20)]
    val_ds   = ChronosFineTuneDataset(
        train_df[train_df[ID_COL].isin(val_ids)], tokenizer,
        context_length=CONTEXT_LENGTH,
    )

    EXTENDED_PATH.mkdir(parents=True, exist_ok=True)
    args = Seq2SeqTrainingArguments(
        output_dir                  = str(EXTENDED_PATH / "checkpoints"),
        max_steps                   = ADDITIONAL_STEPS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        learning_rate               = LEARNING_RATE,
        warmup_steps                = 50,
        lr_scheduler_type           = "cosine",
        eval_strategy               = "steps",
        eval_steps                  = 500,
        save_strategy               = "steps",
        save_steps                  = 500,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        predict_with_generate       = False,
        fp16                        = False,
        bf16                        = torch.cuda.is_available(),
        report_to                   = "none",
        seed                        = RANDOM_SEED,
        logging_steps               = 100,
        dataloader_num_workers      = 0,
    )

    trainer = Seq2SeqTrainer(
        model         = model,
        args          = args,
        train_dataset = train_ds,
        eval_dataset  = val_ds,
        data_collator = default_data_collator,
    )

    print(f"\nExtended fine-tuning ({ADDITIONAL_STEPS} more steps)...")
    trainer.train()

    # Save extended model
    import shutil
    from huggingface_hub import snapshot_download
    from src.finetune_chronos import BASE_MODEL
    base_cache = snapshot_download(BASE_MODEL)
    shutil.copytree(base_cache, str(EXTENDED_PATH), dirs_exist_ok=True)
    model.save_pretrained(str(EXTENDED_PATH))
    print(f"  Saved extended model -> {EXTENDED_PATH}")

    # Evaluate
    print("\nEvaluating extended model...")
    metrics, preds = evaluate_finetuned(EXTENDED_PATH, train_df, test_df)
    agg = metrics[metrics[ID_COL] == "ALL (mean)"].iloc[0]

    preds.to_parquet(MODELS_DIR / "preds_chronos_extended.parquet", index=False)

    # Log to MLflow
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="Chronos-extended"):
        mlflow.log_params({
            "model"           : "Chronos-2",
            "total_steps"     : f"1000+{ADDITIONAL_STEPS}",
            "learning_rate"   : LEARNING_RATE,
        })
        mlflow.log_metrics({k: float(v) for k, v in agg.items()
                             if k != ID_COL and pd.notna(v)})

    print("\n=== CHRONOS FINE-TUNING PROGRESSION ===")
    print(f"  Zero-shot          RMSLE=0.2040  MASE=1.038")
    print(f"  Fine-tuned 1000s   RMSLE=0.1690  MASE=0.863")
    print(f"  Extended {1000+ADDITIONAL_STEPS}s  RMSLE={agg['rmsle']:.4f}  MASE={agg['mase']:.4f}")

    return metrics


if __name__ == "__main__":
    run()
