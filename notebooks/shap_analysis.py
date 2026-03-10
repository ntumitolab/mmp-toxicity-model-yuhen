"""
SHAP feature-importance analysis — marimo reactive notebook.

Launch with:
    uv run marimo edit notebooks/shap_analysis.py
"""

import marimo

__generated_with = "0.4.0"
app = marimo.App()


@app.cell
def _imports():
    import marimo as mo
    import numpy as np
    import torch
    import shap
    import matplotlib.pyplot as plt
    return mo, np, torch, shap, plt


@app.cell
def _controls(mo):
    model_path_input = mo.ui.text(
        value="output/model/model.pth",
        label="Model weights path (.pth)",
    )
    data_dir_input = mo.ui.text(
        value="model/input",
        label="Data directory",
    )
    feature_type_input = mo.ui.dropdown(
        options=["morgan", "macc", "combined", "mordred"],
        value="morgan",
        label="Feature type",
    )
    n_background_input = mo.ui.slider(
        start=50, stop=500, step=50, value=100,
        label="SHAP background samples",
    )
    mo.vstack([model_path_input, data_dir_input, feature_type_input, n_background_input])
    return model_path_input, data_dir_input, feature_type_input, n_background_input


@app.cell
def _load_data(data_dir_input, feature_type_input, mo):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from src.data.loader import load_splits
    from src.data.augment import canonicalize_smiles
    from src.data.featurize import featurize

    try:
        X_train_raw, y_train, X_test_raw, y_test, _, _ = load_splits(
            data_dir_input.value, input_col="Smile"
        )
        X_test_can, y_test_can = canonicalize_smiles(X_test_raw, y_test)
        X_test_np, y_test_np = featurize(X_test_can, y_test_can, feature_type_input.value)
        data_status = mo.callout(mo.md(f"Loaded {len(X_test_np)} test molecules."), kind="success")
    except Exception as e:
        X_test_np, y_test_np = None, None
        data_status = mo.callout(mo.md(f"**Error loading data:** {e}"), kind="danger")

    data_status
    return X_test_np, y_test_np, data_status


@app.cell
def _load_model(model_path_input, feature_type_input, X_test_np, mo, torch):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.model.network import FCModel

    if X_test_np is None:
        model = None
        model_status = mo.callout(mo.md("Data not loaded."), kind="warn")
    else:
        try:
            model = FCModel(input_size=X_test_np.shape[1])
            state = torch.load(model_path_input.value, map_location="cpu")
            model.load_state_dict(state)
            model.eval()
            model_status = mo.callout(mo.md(f"Model loaded: `{model}`"), kind="success")
        except Exception as e:
            model = None
            model_status = mo.callout(mo.md(f"**Error loading model:** {e}"), kind="danger")

    model_status
    return model, model_status


@app.cell
def _shap_explain(model, X_test_np, n_background_input, shap, np, mo, plt):
    if model is None or X_test_np is None:
        mo.callout(mo.md("Model or data not available."), kind="warn")
    else:
        background = X_test_np[
            np.random.choice(len(X_test_np), min(n_background_input.value, len(X_test_np)), replace=False)
        ]

        def predict_fn(x):
            import torch
            with torch.no_grad():
                return model(torch.from_numpy(x).float()).numpy().flatten()

        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X_test_np[:50], nsamples=100)

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test_np[:50], show=False)
        plt.tight_layout()
        mo.pyplot(fig)
