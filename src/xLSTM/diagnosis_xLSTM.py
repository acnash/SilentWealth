#!/usr/bin/env python3
"""
check_predictions_vs_naive.py

Load a predictions CSV produced by the training pipeline (predictions_val.csv or predictions_test.csv),
compute model vs naive baseline metrics, make a small comparison plot, and print human-readable
conclusions.

This version is robust to the two prediction-CSV formats your pipeline might emit:
  - older format: columns include "y_true", "y_pred", "y_naive"
  - residual format: columns include combinations of
        "y_true_res", "y_pred_res", "y_true_abs", "y_pred_abs", "y_naive"
    In the residual format we reconstruct absolute values as:
        y_true_abs = y_true_res + y_naive
        y_pred_abs = y_pred_res + y_naive

Usage:
    python check_predictions_vs_naive.py --pred_csv artifacts/<RUN_ID>/predictions_val.csv
"""
from __future__ import annotations
import os
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict

def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Return Pearson correlation coefficient or nan if not defined."""
    if len(a) < 2:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 1.0 if np.allclose(a, b) else float("nan")
    return float(np.corrcoef(a, b)[0, 1])

def summarize_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_naive: np.ndarray, tol: float=1e-6) -> Dict[str, float]:
    """Compute core metrics and simple interpretation flags."""
    n = len(y_true)
    if n == 0:
        raise ValueError("Empty arrays passed to summarize_metrics()")

    model_abs_err = np.abs(y_pred - y_true)
    naive_abs_err = np.abs(y_naive - y_true)

    model_mae = float(np.mean(model_abs_err))
    naive_mae = float(np.mean(naive_abs_err))

    model_rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    naive_rmse = float(np.sqrt(np.mean((y_naive - y_true) ** 2)))

    corr_model_naive = safe_corr(y_pred, y_naive)
    corr_naive_true = safe_corr(y_naive, y_true)
    corr_model_true = safe_corr(y_pred, y_true)

    frac_model_equals_naive = float(np.mean(np.isclose(y_pred, y_naive, atol=tol)))

    if math.isfinite(naive_mae) and naive_mae > 0:
        rel_improvement_pct = 100.0 * (naive_mae - model_mae) / naive_mae
    else:
        rel_improvement_pct = float("nan")

    return {
        "n": n,
        "model_mae": model_mae,
        "naive_mae": naive_mae,
        "model_rmse": model_rmse,
        "naive_rmse": naive_rmse,
        "corr_model_naive": corr_model_naive,
        "corr_naive_true": corr_naive_true,
        "corr_model_true": corr_model_true,
        "frac_model_equals_naive": frac_model_equals_naive,
        "rel_improvement_pct": rel_improvement_pct
    }

def print_conclusions(m: Dict[str, float], tol: float=1e-6) -> None:
    """Print human-friendly conclusions based on metric dictionary m."""
    print("\n=== Summary Metrics ===")
    print(f"Samples: {m['n']}")
    print(f"Model   MAE: {m['model_mae']:.6f}")
    print(f"Naive   MAE: {m['naive_mae']:.6f}")
    print(f"Model  RMSE: {m['model_rmse']:.6f}")
    print(f"Naive  RMSE: {m['naive_rmse']:.6f}")
    print(f"Corr(model, naive): {m['corr_model_naive']:.4f}")
    print(f"Corr(naive, true) : {m['corr_naive_true']:.4f}")
    print(f"Corr(model, true) : {m['corr_model_true']:.4f}")
    print(f"Frac predictions ≈ naive (tol={tol}): {m['frac_model_equals_naive']:.4f}")
    if math.isfinite(m['rel_improvement_pct']):
        print(f"Relative MAE improvement vs naive: {m['rel_improvement_pct']:.2f} %")
    else:
        print("Relative MAE improvement vs naive: N/A")

    # Interpretations
    print("\n=== Conclusions (automated) ===")
    # Strength of naive
    if not math.isnan(m['corr_naive_true']) and m['corr_naive_true'] >= 0.80:
        print("- The naive last-value predictor is VERY STRONG (corr >= 0.80).")
        print("  Expect a steep early drop in training loss and that beating naive is difficult.")
    elif not math.isnan(m['corr_naive_true']) and m['corr_naive_true'] >= 0.50:
        print("- The naive predictor is fairly strong (corr >= 0.50).")
    else:
        print("- The naive predictor is weak or low correlation with truth.")

    # Model vs naive performance
    if m['model_mae'] < m['naive_mae']:
        if m['rel_improvement_pct'] >= 10.0:
            print(f"- The model BEATS naive by a meaningful margin ({m['rel_improvement_pct']:.2f}% MAE reduction). Good sign.")
        elif m['rel_improvement_pct'] >= 1.0:
            print(f"- The model slightly improves over naive ({m['rel_improvement_pct']:.2f}% MAE reduction).")
        else:
            print("- The model's improvement over naive is marginal (<1%). Might not be practically useful.")
    elif m['model_mae'] > m['naive_mae']:
        pct_worse = 100.0 * (m['model_mae'] - m['naive_mae']) / (m['naive_mae'] if m['naive_mae']>0 else 1.0)
        print(f"- The model performs WORSE than naive by {pct_worse:.2f}% MAE. Check for issues or consider different target/loss/features.")
    else:
        print("- Model MAE equals naive MAE within machine precision.")

    # Did model simply copy naive?
    if m['frac_model_equals_naive'] >= 0.5:
        print("- Over half of the model's predictions are essentially equal to the naive last-value (within tolerance).")
        print("  This suggests the model has learned to mimic the naive rule in many places.")
    elif m['corr_model_naive'] >= 0.9:
        print("- High correlation between model predictions and naive predictions (>0.9). Possible mimicry.")
    else:
        print("- The model is not simply reproducing the naive prediction everywhere.")

    # Additional pointers
    if m['model_mae'] >= m['naive_mae'] and not math.isnan(m['corr_naive_true']) and m['corr_naive_true'] >= 0.5:
        print("\nHint: Consider predicting returns (y_t / y_{t-1} - 1 or log returns) or the residual (y_t - y_{t-1})")
        print("      rather than raw price. This reduces dominance of the last-value baseline and focuses learning on change.")
    print("")  # blank line

def plot_comparison(df: pd.DataFrame, out_path: str, n_points: int = 300) -> Optional[str]:
    """Plot first n_points of y_true, y_pred, y_naive and save to out_path."""
    n = min(n_points, len(df))
    if n == 0:
        return None
    sub = df.iloc[:n]
    plt.figure(figsize=(12, 3))
    plt.plot(sub["y_true"].values, "-k", linewidth=1.0, label="y_true")
    plt.plot(sub["y_pred"].values, "-r", linewidth=1.0, label="y_pred")
    plt.plot(sub["y_naive"].values, "--", linewidth=1.0, label="y_naive")
    plt.title(f"Comparison (first {n} samples)")
    plt.xlabel("Window index")
    plt.ylabel("Scaled value")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path

def per_ticker_summary(df: pd.DataFrame, top_k: int = 5):
    """If 'Ticker' exists, compute MAE per ticker for model and naive and show top/bottom improvements."""
    if "Ticker" not in df.columns:
        return None
    rows = []
    for tkr, g in df.groupby("Ticker"):
        y_true = g["y_true"].to_numpy()
        y_pred = g["y_pred"].to_numpy()
        y_naive = g["y_naive"].to_numpy()
        if len(y_true) == 0:
            continue
        model_mae = float(np.mean(np.abs(y_pred - y_true)))
        naive_mae = float(np.mean(np.abs(y_naive - y_true)))
        rel_imp = (naive_mae - model_mae) / (naive_mae if naive_mae>0 else 1.0)
        rows.append((tkr, model_mae, naive_mae, rel_imp))
    if not rows:
        return None
    df_t = pd.DataFrame(rows, columns=["Ticker","model_mae","naive_mae","rel_imp"])
    df_t["rel_imp_pct"] = df_t["rel_imp"] * 100.0
    df_t = df_t.sort_values("rel_imp_pct", ascending=False)
    best = df_t.head(top_k)
    worst = df_t.tail(top_k).sort_values("rel_imp_pct")
    return best, worst, df_t

def find_default_pred_file() -> Optional[str]:
    """Try common defaults (val then test) in 'artifacts/*' if user didn't supply path."""
    # Try local files first
    for fname in ("predictions_val.csv", "predictions_test.csv"):
        if os.path.exists(fname):
            return fname
    if not os.path.isdir("artifacts"):
        return None
    # Walk artifacts
    for root, dirs, files in os.walk("artifacts"):
        if "predictions_val.csv" in files:
            return os.path.join(root, "predictions_val.csv")
    for root, dirs, files in os.walk("artifacts"):
        if "predictions_test.csv" in files:
            return os.path.join(root, "predictions_test.csv")
    return None

def load_and_normalize_preds(path: str) -> Tuple[pd.DataFrame, str]:
    """
    Load the CSV and return a DataFrame normalized to columns:
       - "y_true" (absolute)
       - "y_pred" (absolute)
       - "y_naive" (absolute)
    Returns (df_norm, notes) where notes is a short string explaining what transformations were done.
    Raises ValueError if required columns cannot be constructed.
    """
    df = pd.read_csv(path)
    notes = []
    # If already contains canonical names, use them
    if {"y_true","y_pred","y_naive"}.issubset(df.columns):
        notes.append("Using existing columns: y_true, y_pred, y_naive")
        df_norm = df.copy()
        # Ensure numeric
        df_norm = df_norm.replace([np.inf, -np.inf], np.nan)
        return df_norm, "; ".join(notes)

    # If absolute names exist from new pipeline
    if "y_true_abs" in df.columns or "y_pred_abs" in df.columns:
        # y_naive must exist to be meaningful (it's the last input)
        if "y_naive" not in df.columns:
            raise ValueError("CSV contains y_true_abs/y_pred_abs but missing y_naive; cannot normalize.")
        notes.append("Found y_true_abs/y_pred_abs + y_naive; mapping to canonical columns.")
        y_true = df["y_true_abs"].to_numpy() if "y_true_abs" in df.columns else None
        y_pred = df["y_pred_abs"].to_numpy() if "y_pred_abs" in df.columns else None
        y_naive = df["y_naive"].to_numpy()
        df_norm = df.copy()
        if y_true is not None:
            df_norm["y_true"] = y_true
        if y_pred is not None:
            df_norm["y_pred"] = y_pred
        df_norm["y_naive"] = y_naive
        return df_norm, "; ".join(notes)

    # If residual format present: y_true_res/y_pred_res and y_naive present -> reconstruct
    if ("y_true_res" in df.columns or "y_pred_res" in df.columns) and "y_naive" in df.columns:
        notes.append("Found residual columns (y_*_res) + y_naive; reconstructing absolute values.")
        y_naive = df["y_naive"].to_numpy()
        # broadcast/align shapes carefully (if y_naive is length N it's fine)
        df_norm = df.copy()
        if "y_true_res" in df.columns:
            df_norm["y_true"] = df["y_true_res"].to_numpy() + y_naive
        if "y_pred_res" in df.columns:
            df_norm["y_pred"] = df["y_pred_res"].to_numpy() + y_naive
        df_norm["y_naive"] = y_naive
        return df_norm, "; ".join(notes)

    # Another fallback: if the file uses y_true_abs and only y_pred_res, we can reconstruct y_pred_abs:
    if "y_true_abs" in df.columns and "y_pred_res" in df.columns and "y_naive" in df.columns:
        notes.append("Mixed columns: reconstructing y_pred_abs from y_pred_res + y_naive; using existing y_true_abs.")
        df_norm = df.copy()
        df_norm["y_true"] = df["y_true_abs"]
        df_norm["y_pred"] = df["y_pred_res"].to_numpy() + df["y_naive"].to_numpy()
        df_norm["y_naive"] = df["y_naive"]
        return df_norm, "; ".join(notes)

    # If none of the above match, fail with helpful message
    raise ValueError(
        "Could not find required columns in CSV. Expected either:\n"
        "  - y_true, y_pred, y_naive  (old format)\n"
        "  - y_true_abs / y_pred_abs and y_naive  (absolute new format)\n"
        "  - y_true_res / y_pred_res and y_naive  (residual new format, reconstructable)\n"
        f"Available columns: {list(df.columns)}"
    )

def main():
    p = argparse.ArgumentParser(description="Check model predictions vs naive baseline and print conclusions.")
    p.add_argument("--pred_csv", type=str, default="", help="Path to predictions CSV (predictions_val.csv or predictions_test.csv).")
    p.add_argument("--tol", type=float, default=1e-6, help="Tolerance for checking prediction ≈ naive (default 1e-6).")
    p.add_argument("--plot_n", type=int, default=300, help="Number of samples to plot from the start (default 300).")
    args = p.parse_args()

    pred_csv = args.pred_csv.strip()
    if pred_csv == "":
        pred_csv = find_default_pred_file()
        if pred_csv is None:
            print("No --pred_csv provided and no default predictions file found under ./artifacts/.")
            print("Please pass --pred_csv path to predictions_val.csv or predictions_test.csv")
            return
        print(f"No --pred_csv provided, using discovered file: {pred_csv}")

    if not os.path.exists(pred_csv):
        print(f"Predictions file not found: {pred_csv}")
        return

    try:
        df_norm, notes = load_and_normalize_preds(pred_csv)
    except ValueError as e:
        print(f"Error loading predictions CSV: {e}")
        return

    print(f"Loaded predictions from: {pred_csv}")
    print("Normalization notes:", notes)

    # Drop rows with NaNs in the core canonical columns
    required = {"y_true","y_pred","y_naive"}
    missing = required - set(df_norm.columns)
    if missing:
        print(f"After normalization, required columns missing: {missing}")
        return

    df_norm = df_norm.replace([np.inf, -np.inf], np.nan)
    df_norm = df_norm.dropna(subset=["y_true","y_pred","y_naive"]).reset_index(drop=True)
    if len(df_norm) == 0:
        print("No valid rows remaining after dropping NaNs.")
        return

    # Print a tiny header sample for sanity
    print("\nFirst 5 rows (sanity check):")
    print(df_norm[["y_true","y_pred","y_naive"]].head().to_string(index=False, float_format="%.6f"))

    y_true = df_norm["y_true"].to_numpy(dtype=float)
    y_pred = df_norm["y_pred"].to_numpy(dtype=float)
    y_naive = df_norm["y_naive"].to_numpy(dtype=float)

    metrics = summarize_metrics(y_true, y_pred, y_naive, tol=args.tol)
    print_conclusions(metrics, tol=args.tol)

    # Save comparison plot
    out_plot = os.path.join(os.path.dirname(pred_csv) or ".", "comparison_first_samples.png")
    # for plotting we need canonical columns in the df
    plot_df = df_norm.copy()
    # rename to ensure plot_comparison finds the columns
    plot_df = plot_df.rename(columns={"y_true":"y_true","y_pred":"y_pred","y_naive":"y_naive"})
    saved = plot_comparison(plot_df, out_plot, n_points=args.plot_n)
    if saved:
        print(f"Comparison plot saved to: {saved}")
    else:
        print("Not enough points to plot comparison.")

    # Per-ticker summary if available
    per = per_ticker_summary(df_norm, top_k=5)
    if per is not None:
        best, worst, df_t = per
        print("\nTop 5 tickers where model improved most vs naive (percent):")
        print(best[["Ticker","model_mae","naive_mae","rel_imp_pct"]].to_string(index=False, float_format='%.3f'))
        print("\nTop 5 tickers where model did WORST vs naive (most negative improvement):")
        print(worst[["Ticker","model_mae","naive_mae","rel_imp_pct"]].to_string(index=False, float_format='%.3f'))
        per_path = os.path.join(os.path.dirname(pred_csv) or ".", "per_ticker_mae_summary.csv")
        df_t.to_csv(per_path, index=False)
        print(f"\nPer-ticker MAE table saved to: {per_path}")

if __name__ == "__main__":
    main()
