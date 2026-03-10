"""
PCA / t-SNE chemical-space visualisation — marimo reactive notebook.

Projects molecular fingerprints into 2D and colours points by class label,
letting you visually assess how well the feature space separates toxic from
non-toxic molecules.  Also plots the cumulative explained-variance curve so
you can read off how many PCA components capture 90 % of the variance.

Launch with:
    uv run marimo edit notebooks/pca_analysis.py
"""

import marimo

__generated_with = "0.20.2"
app = marimo.App()


@app.cell
def _imports():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell
def _controls(mo):
    data_dir_input = mo.ui.text(value="input", label="Data directory")
    feature_type_input = mo.ui.dropdown(
        options=["morgan", "macc", "combined", "mordred", "pubchem"],
        value="morgan",
        label="Feature type",
    )
    split_input = mo.ui.dropdown(
        options=["random", "scaffold"],
        value="random",
        label="Split mode (which molecules to show)",
    )
    tsne_perplexity_input = mo.ui.slider(
        start=5, stop=80, step=5, value=30,
        label="t-SNE perplexity",
    )
    max_sample_input = mo.ui.slider(
        start=100, stop=2000, step=100, value=500,
        label="Max molecules to project (t-SNE speed)",
    )
    mo.vstack([
        data_dir_input, feature_type_input, split_input,
        tsne_perplexity_input, max_sample_input,
    ])
    return (
        data_dir_input,
        feature_type_input,
        max_sample_input,
        split_input,
        tsne_perplexity_input,
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
            X_tr_raw, y_tr, X_te_raw, y_te, _, _, detected_cols = load_splits(
                data_dir_input.value, input_col="Inchi"
            )
            def _conv(X, y, col):
                return inchi_to_smiles(X, y) if col == "Inchi" else canonicalize_smiles(X, y)
            X_tr, y_tr = _conv(X_tr_raw, y_tr, detected_cols["Train"])
            X_te, y_te = _conv(X_te_raw, y_te, detected_cols["Test"])
            X_all = X_tr + X_te
            y_all = y_tr + y_te
            split_label = ["Train"] * len(X_tr) + ["Test"] * len(X_te)
        else:
            from src.data.loader import load_combined
            from src.data.scaffold_split import scaffold_split
            X_tr_raw, y_tr_raw, X_te_raw, y_te_raw, _, _, detected_cols = (
                load_combined(data_dir_input.value)
            )
            def _conv2(X, y, col):
                return inchi_to_smiles(X, y) if col == "Inchi" else canonicalize_smiles(X, y)
            X_tr, y_tr = _conv2(X_tr_raw, y_tr_raw, detected_cols["Train"])
            X_te, y_te = _conv2(X_te_raw, y_te_raw, detected_cols["Test"])
            X_pool = X_tr + X_te
            y_pool = y_tr + y_te
            X_sc_tr, y_sc_tr, X_sc_te, y_sc_te = scaffold_split(
                X_pool, y_pool, test_frac=0.12
            )
            X_all = X_sc_tr + X_sc_te
            y_all = y_sc_tr + y_sc_te
            split_label = ["Train"] * len(X_sc_tr) + ["Test"] * len(X_sc_te)

        X_can, y_can = canonicalize_smiles(X_all, y_all)
        split_label_can = [split_label[i] for i, (x, _) in
                           enumerate(zip(X_all, y_all))
                           if x in set(X_can)][:len(X_can)]

        X_feat, y_np = featurize(X_can, y_can, feature_type_input.value)
        status = mo.callout(
            mo.md(f"Featurised **{X_feat.shape[0]}** molecules, "
                  f"**{X_feat.shape[1]}** features."),
            kind="success",
        )
    except Exception as e:
        X_feat = y_np = split_label_can = None
        status = mo.callout(mo.md(f"**Error:** {e}"), kind="danger")

    status
    return X_feat, split_label_can, y_np


@app.cell
def _pca_variance(X_feat, mo, np, plt):
    mo.stop(X_feat is None, None)

    from sklearn.decomposition import PCA

    pca_full = PCA(random_state=42)
    pca_full.fit(X_feat)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n90 = int(np.searchsorted(cum_var, 0.90)) + 1
    n95 = int(np.searchsorted(cum_var, 0.95)) + 1

    def _draw():
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(range(1, len(cum_var) + 1), cum_var, lw=1.5, color="steelblue")
        for thresh, n_comp, c in [(0.90, n90, "#e07070"), (0.95, n95, "#70b070")]:
            ax.axhline(thresh, color=c, linestyle="--", linewidth=1,
                       label=f"{int(thresh*100)}% variance → {n_comp} components")
            ax.axvline(n_comp, color=c, linestyle="--", linewidth=1)
        ax.set_xlabel("Number of PCA components")
        ax.set_ylabel("Cumulative explained variance")
        ax.set_title("PCA — Cumulative Explained Variance")
        ax.legend()
        plt.tight_layout()
        return mo.vstack([
            fig,
            mo.callout(
                mo.md(
                    f"**{n90}** components explain 90% of variance  "
                    f"(out of {X_feat.shape[1]} features)  \n"
                    f"**{n95}** components explain 95% of variance"
                ),
                kind="info",
            ),
        ])

    _draw()
    return


@app.cell
def _pca_2d(X_feat, mo, np, plt, split_label_can, y_np):
    mo.stop(X_feat is None, None)

    def _run():
        from sklearn.decomposition import PCA

        pca2 = PCA(n_components=2, random_state=42)
        coords = pca2.fit_transform(X_feat)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax0 = axes[0]
        for lbl, col, nm in [(0, "#4a90d9", "Non-toxic"), (1, "#e05050", "Toxic")]:
            mask = y_np == lbl
            ax0.scatter(coords[mask, 0], coords[mask, 1],
                        c=col, alpha=0.35, s=8, label=nm)
        ax0.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)")
        ax0.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)")
        ax0.set_title("PCA — coloured by class")
        ax0.legend(markerscale=3)

        ax1 = axes[1]
        if split_label_can is not None:
            labels_arr = np.array(split_label_can[:len(coords)])
            for sp, col in [("Train", "#5aabbb"), ("Test", "#e09030")]:
                mask = labels_arr == sp
                ax1.scatter(coords[mask, 0], coords[mask, 1],
                            c=col, alpha=0.35, s=8, label=sp)
            ax1.set_title("PCA — coloured by split")
            ax1.legend(markerscale=3)
        else:
            ax1.set_visible(False)
        ax1.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)")
        ax1.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)")

        plt.tight_layout()
        return fig

    _run()
    return


@app.cell
def _tsne_2d(
    X_feat,
    max_sample_input,
    mo,
    np,
    plt,
    split_label_can,
    tsne_perplexity_input,
    y_np,
):
    mo.stop(X_feat is None, None)

    def _run():
        from sklearn.manifold import TSNE

        n_pts = min(max_sample_input.value, len(X_feat))
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_feat), size=n_pts, replace=False)
        X_sub  = X_feat[idx]
        y_sub  = y_np[idx]
        sp_sub = (np.array(split_label_can)[idx]
                  if split_label_can is not None else None)

        coords = TSNE(
            n_components=2,
            perplexity=tsne_perplexity_input.value,
            random_state=42,
            n_jobs=-1,
        ).fit_transform(X_sub)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax0 = axes[0]
        for lbl, col, nm in [(0, "#4a90d9", "Non-toxic"), (1, "#e05050", "Toxic")]:
            mask = y_sub == lbl
            ax0.scatter(coords[mask, 0], coords[mask, 1],
                        c=col, alpha=0.4, s=8, label=nm)
        ax0.set_title(
            f"t-SNE (n={n_pts}, perplexity={tsne_perplexity_input.value}) — class"
        )
        ax0.set_xlabel("t-SNE 1")
        ax0.set_ylabel("t-SNE 2")
        ax0.legend(markerscale=3)

        ax1 = axes[1]
        if sp_sub is not None:
            for sp, col in [("Train", "#5aabbb"), ("Test", "#e09030")]:
                mask = sp_sub == sp
                ax1.scatter(coords[mask, 0], coords[mask, 1],
                            c=col, alpha=0.4, s=8, label=sp)
            ax1.set_title("t-SNE — split")
            ax1.legend(markerscale=3)
        else:
            ax1.set_visible(False)
        ax1.set_xlabel("t-SNE 1")
        ax1.set_ylabel("t-SNE 2")

        plt.tight_layout()
        return fig

    _run()
    return


if __name__ == "__main__":
    app.run()
