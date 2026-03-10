"""
Classical ML model comparison — marimo reactive notebook.

Trains Random Forest, Gradient Boosting, Extra Trees, SVM, CatBoost and KNN
on the same featurisation pipeline used by the deep-learning model, so you can
put all results in one table for the paper.

Supports both random-split and scaffold-split modes.
Results are saved to output/table/ml_results_{split}_{feature}.xlsx.

Launch with:
    uv run marimo edit notebooks/ml_models.py
"""

import marimo

__generated_with = "0.20.2"
app = marimo.App()


@app.cell
def _imports():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    return mo, np, pd, plt


@app.cell
def _controls(mo):
    data_dir_input = mo.ui.text(value="input", label="Data directory")
    output_dir_input = mo.ui.text(value="output", label="Output directory")
    feature_type_input = mo.ui.dropdown(
        options=["morgan", "macc", "combined", "mordred", "pubchem"],
        value="morgan",
        label="Feature type",
    )
    split_input = mo.ui.dropdown(
        options=["random", "scaffold"],
        value="random",
        label="Split mode",
    )
    n_folds_input = mo.ui.slider(
        start=3, stop=10, step=1, value=5,
        label="CV folds",
    )
    n_iter_input = mo.ui.slider(
        start=10, stop=100, step=10, value=10,
        label="RandomizedSearchCV iterations",
    )
    run_button = mo.ui.run_button(label="▶ Train all models")
    mo.vstack([
        data_dir_input, output_dir_input,
        feature_type_input, split_input,
        n_folds_input, n_iter_input, run_button,
    ])
    return (
        data_dir_input,
        feature_type_input,
        n_folds_input,
        n_iter_input,
        output_dir_input,
        run_button,
        split_input,
    )


@app.cell
def _load_featurize(data_dir_input, feature_type_input, mo, split_input):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from src.data.augment import canonicalize_smiles, inchi_to_smiles
    from src.data.featurize import featurize

    try:
        if split_input.value == "random":
            from src.data.loader import load_splits
            X_tr_raw, y_tr_raw, X_te_raw, y_te_raw, _, _, detected_cols = load_splits(
                data_dir_input.value, input_col="Inchi"
            )
            def _conv(X, y, col):
                return inchi_to_smiles(X, y) if col == "Inchi" else canonicalize_smiles(X, y)
            X_tr_s, y_tr_s = _conv(X_tr_raw, y_tr_raw, detected_cols["Train"])
            X_te_s, y_te_s = _conv(X_te_raw, y_te_raw, detected_cols["Test"])

        else:  # scaffold
            from src.data.loader import load_combined
            from src.data.scaffold_split import scaffold_split
            X_tr_raw, y_tr_raw, X_te_raw, y_te_raw, _, _, detected_cols = (
                load_combined(data_dir_input.value)
            )
            def _conv2(X, y, col):
                return inchi_to_smiles(X, y) if col == "Inchi" else canonicalize_smiles(X, y)
            X_tr_c, y_tr_c = _conv2(X_tr_raw, y_tr_raw, detected_cols["Train"])
            X_te_c, y_te_c = _conv2(X_te_raw, y_te_raw, detected_cols["Test"])
            X_pool = X_tr_c + X_te_c
            y_pool = y_tr_c + y_te_c
            X_tr_s, y_tr_s, X_te_s, y_te_s = scaffold_split(
                X_pool, y_pool, test_frac=0.12
            )

        X_tr_can, y_tr_can = canonicalize_smiles(X_tr_s, y_tr_s)
        X_te_can, y_te_can = canonicalize_smiles(X_te_s, y_te_s)

        X_train, y_train = featurize(X_tr_can, y_tr_can, feature_type_input.value)
        X_test,  y_test  = featurize(X_te_can, y_te_can, feature_type_input.value)

        status = mo.callout(
            mo.md(
                f"**{split_input.value.capitalize()} split** — "
                f"train: {X_train.shape[0]} × {X_train.shape[1]},  "
                f"test: {X_test.shape[0]} × {X_test.shape[1]}"
            ),
            kind="success",
        )
    except Exception as e:
        X_train = y_train = X_test = y_test = None
        status = mo.callout(mo.md(f"**Error loading data:** {e}"), kind="danger")

    status
    return X_test, X_train, y_test, y_train


@app.cell
def _train_models(
    X_test,
    X_train,
    feature_type_input,
    mo,
    n_folds_input,
    n_iter_input,
    np,
    output_dir_input,
    pd,
    run_button,
    split_input,
    y_test,
    y_train,
):
    # np, pd injected from _imports — no top-level imports needed here
    mo.stop(
        not run_button.value,
        mo.callout(mo.md("Press **▶ Train all models** to start."), kind="neutral"),
    )
    mo.stop(
        X_train is None,
        mo.callout(mo.md("Data not loaded."), kind="warn"),
    )

    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.metrics import (
        f1_score, matthews_corrcoef, roc_auc_score,
        balanced_accuracy_score, confusion_matrix, make_scorer,
    )
    from catboost import CatBoostClassifier

    # Scoring: Yuhen's combined (MCC + F1) / 2  (random_state=50 matches original)
    def _mcc_f1(y_true, y_pred):
        return (matthews_corrcoef(y_true, y_pred) + f1_score(y_true, y_pred)) / 2

    scoring = {
        "Combined": make_scorer(_mcc_f1),
        "F1":       "f1",
        "MCC":      make_scorer(matthews_corrcoef),
    }

    # Classifiers and param grids match Yuhen's original notebooks exactly.
    # CatBoost note: RMSE/MAE are regression losses; those CV iterations will
    # produce nan scores (caught by error_score=np.nan) and be skipped.
    classifiers = {
        "Random Forest": (
            Pipeline([("classifier", RandomForestClassifier())]),
            {
                "classifier__max_features":  ["sqrt", "log2", None],
                "classifier__max_depth":     range(10, 1001, 20),
                "classifier__n_estimators":  range(100, 2001, 100),
                "classifier__criterion":     ["gini", "entropy"],
                "classifier__oob_score":     [True, False],
                "classifier__class_weight":  ["balanced", None],
            },
        ),
        "KNN": (
            Pipeline([("classifier", KNeighborsClassifier())]),
            {
                "classifier__n_neighbors": range(1, 50),
                "classifier__weights":     ["uniform", "distance"],
                "classifier__algorithm":   ["auto", "ball_tree", "kd_tree", "brute"],
                "classifier__leaf_size":   range(10, 101, 10),
            },
        ),
        "Radius Neighbors": (
            Pipeline([("classifier", RadiusNeighborsClassifier(outlier_label="most_frequent"))]),
            {
                "classifier__radius":    np.arange(1.0, 10.0, 0.5),
                "classifier__weights":   ["uniform", "distance"],
                "classifier__algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                "classifier__leaf_size": range(10, 101, 10),
            },
        ),
        "SVM": (
            Pipeline([("classifier", SVC())]),
            {
                "classifier__kernel":       ["linear", "rbf", "sigmoid", "poly"],
                "classifier__C":            [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 500, 1000],
                "classifier__gamma":        ["scale", "auto", 0.1, 1.0, 5.0, 10.0],
                "classifier__class_weight": ["balanced"],
                "classifier__probability":  [True],
            },
        ),
        "MLP": (
            Pipeline([("classifier", MLPClassifier())]),
            {
                "classifier__hidden_layer_sizes": [50, 100, 150, 200, 300, 400],
                "classifier__solver":             ["sgd", "adam"],
                "classifier__learning_rate_init": [0.001, 0.01, 0.015, 0.025, 0.05, 0.1, 0.5],
                "classifier__early_stopping":     [True],
                "classifier__max_iter":           [500],
            },
        ),
        "CatBoost": (
            Pipeline([("classifier", CatBoostClassifier(verbose=0))]),
            {
                "classifier__loss_function":      ["RMSE", "Logloss", "CrossEntropy", "MAE"],
                "classifier__iterations":         [50, 100, 500, 1000, 1500, 2000],
                "classifier__learning_rate":      [0.001, 0.01, 0.015, 0.025, 0.05, 0.1, 0.5],
                "classifier__sampling_frequency": ["PerTree", "PerTreeLevel"],
                "classifier__silent":             [True],
            },
        ),
    }

    def _metrics(y_true, y_pred, y_prob=None):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        result = {
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "F1":            round(f1_score(y_true, y_pred), 4),
            "MCC":           round(matthews_corrcoef(y_true, y_pred), 4),
            "Sensitivity":   round(tp / max(tp + fn, 1), 4),
            "Recall":        round(tp / max(tp + fn, 1), 4),
            "Precision":     round(tp / max(tp + fp, 1), 4),
            "Specificity":   round(tn / max(tn + fp, 1), 4),
            "Accuracy":      round((tp + tn) / max(tp + tn + fp + fn, 1), 4),
            "Bal. Accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        }
        if y_prob is not None:
            result["AUC-ROC"] = round(roc_auc_score(y_true, y_prob), 4)
        return result

    rows_cv, rows_test = [], []

    for name, (pipeline, param_grid) in classifiers.items():
        search = RandomizedSearchCV(
            pipeline,
            param_grid,
            n_iter=n_iter_input.value,
            scoring=scoring,
            refit="Combined",
            cv=n_folds_input.value,
            random_state=50,
            n_jobs=-1,
            error_score=np.nan,
        )
        search.fit(X_train, y_train)

        best_idx  = search.best_index_
        comb_mean = search.cv_results_["mean_test_Combined"][best_idx]
        comb_std  = search.cv_results_["std_test_Combined"][best_idx]
        f1_mean   = search.cv_results_["mean_test_F1"][best_idx]
        f1_std    = search.cv_results_["std_test_F1"][best_idx]
        mcc_mean  = search.cv_results_["mean_test_MCC"][best_idx]
        mcc_std   = search.cv_results_["std_test_MCC"][best_idx]

        rows_cv.append({
            "Model":                    name,
            "CV (MCC+F1)/2 (mean ± SD)": f"{comb_mean:.4f} ± {comb_std:.4f}",
            "CV F1 (mean ± SD)":         f"{f1_mean:.4f} ± {f1_std:.4f}",
            "CV MCC (mean ± SD)":        f"{mcc_mean:.4f} ± {mcc_std:.4f}",
            "Best params":               str(search.best_params_),
        })

        best_est = search.best_estimator_
        y_pred_t = best_est.predict(X_test)
        y_prob_t = (best_est.predict_proba(X_test)[:, 1]
                    if hasattr(best_est, "predict_proba") else None)
        row = _metrics(y_test, y_pred_t, y_prob_t)
        row["Model"] = name
        rows_test.append(row)

    df_cv   = pd.DataFrame(rows_cv)
    df_test = pd.DataFrame(rows_test).set_index("Model")

    # os is used only inside this nested function so the static analyser
    # doesn't flag it as a cell-level name (avoiding conflict with _load_featurize)
    def _save_excel(out_dir, split_name, feature_name):
        import os
        os.makedirs(os.path.join(out_dir, "table"), exist_ok=True)
        path = os.path.join(
            out_dir, "table",
            f"ml_results_{split_name}_{feature_name}.xlsx",
        )
        with pd.ExcelWriter(path) as writer:
            df_cv.to_excel(writer,   sheet_name="Cross-validation", index=False)
            df_test.to_excel(writer, sheet_name="Test metrics")
        return path

    out_path = _save_excel(
        output_dir_input.value, split_input.value, feature_type_input.value
    )
    mo.callout(
        mo.md(f"Training complete. Results saved to `{out_path}`."),
        kind="success",
    )
    return df_cv, df_test


@app.cell
def _cv_table(df_cv, mo):
    mo.stop(df_cv is None, None)
    mo.vstack([
        mo.md("### Cross-validation results (train set)"),
        mo.ui.table(df_cv),
    ])
    return


@app.cell
def _test_table(df_test, mo):
    mo.stop(df_test is None, None)
    mo.vstack([
        mo.md("### Test-set metrics"),
        mo.ui.table(df_test.reset_index()),
    ])
    return


@app.cell
def _comparison_bar(df_test, mo, np, plt):
    # np, plt injected from _imports
    mo.stop(df_test is None, None)

    models   = df_test.index.tolist()
    f1_vals  = df_test["F1"].values
    mcc_vals = df_test["MCC"].values
    x = np.arange(len(models))
    w = 0.38

    fig, ax = plt.subplots(figsize=(max(9, len(models) * 1.4), 5))
    ax.bar(x - w / 2, f1_vals,  w, label="F1",  color="#5aabbb", alpha=0.85)
    ax.bar(x + w / 2, mcc_vals, w, label="MCC", color="#e09030", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_title("ML Models — Test-set F1 and MCC")
    ax.legend()
    for i, (f1, mcc) in enumerate(zip(f1_vals, mcc_vals)):
        ax.text(i - w / 2, f1  + 0.015, f"{f1:.3f}",  ha="center", fontsize=8)
        ax.text(i + w / 2, mcc + 0.015, f"{mcc:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    fig
    return


@app.cell
def _radar_chart(df_test, mo, np, plt):
    # np, plt injected from _imports
    mo.stop(df_test is None, None)

    radar_metrics = ["F1", "MCC", "Sensitivity", "Precision", "Specificity", "Bal. Accuracy"]
    available = [m for m in radar_metrics if m in df_test.columns]
    mo.stop(len(available) < 3, None)

    N = len(available)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig2, ax2 = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10.colors

    for j, (model, test_row) in enumerate(df_test.iterrows()):
        vals = [test_row[m] for m in available] + [test_row[available[0]]]
        ax2.plot(angles, vals, "o-", linewidth=1.5,
                 color=colors[j % len(colors)], label=model)
        ax2.fill(angles, vals, alpha=0.08, color=colors[j % len(colors)])

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(available, size=10)
    ax2.set_ylim(0, 1)
    ax2.set_title("ML Models — Metric Radar (test set)", pad=20)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    plt.tight_layout()
    fig2
    return


if __name__ == "__main__":
    app.run()
