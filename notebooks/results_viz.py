"""
Cross-validation results visualisation — marimo reactive notebook.

Launch with:
    uv run marimo edit notebooks/results_viz.py
"""

import marimo

__generated_with = "0.4.0"
app = marimo.App()


@app.cell
def _imports():
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    return mo, pd, plt, np


@app.cell
def _controls(mo):
    cv_path_input = mo.ui.text(
        value="output/table/cross_validation.xlsx",
        label="cross_validation.xlsx path",
    )
    cv_path_input
    return (cv_path_input,)


@app.cell
def _load_cv(cv_path_input, mo, pd):
    try:
        df = pd.read_excel(cv_path_input.value)
        status = mo.callout(mo.md(f"Loaded {len(df)} folds."), kind="success")
    except Exception as e:
        df = None
        status = mo.callout(mo.md(f"**Error:** {e}"), kind="danger")

    status
    return (df,)


@app.cell
def _table(df, mo):
    if df is not None:
        mo.ui.table(df)
    else:
        mo.md("No data.")


@app.cell
def _bar_chart(df, mo, plt, np):
    if df is None:
        mo.md("No data to plot.")
    else:
        metric_cols = ["Training F1", "Validation F1", "Testing F1", "Testing 2 F1",
                       "Training MCC", "Validation MCC", "Testing MCC", "Testing 2 MCC"]
        available = [c for c in metric_cols if c in df.columns]
        means = df[available].mean()
        stds  = df[available].std()

        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(available))
        bars = ax.bar(x, means, yerr=stds, capsize=4, color="steelblue", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(available, rotation=30, ha="right")
        ax.set_ylabel("Score")
        ax.set_title("Mean ± SD across folds")
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        mo.pyplot(fig)


@app.cell
def _per_fold_lines(df, mo, plt):
    if df is None:
        mo.md("No data.")
    else:
        f1_cols = [c for c in ["Training F1", "Validation F1", "Testing F1", "Testing 2 F1"] if c in df.columns]
        fig, ax = plt.subplots(figsize=(8, 4))
        for col in f1_cols:
            ax.plot(df["Fold"], df[col], marker="o", label=col)
        ax.set_xlabel("Fold")
        ax.set_ylabel("F1")
        ax.set_title("F1 per fold")
        ax.legend()
        plt.tight_layout()
        mo.pyplot(fig)
