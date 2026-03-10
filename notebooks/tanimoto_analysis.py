"""
Tanimoto chemical-similarity analysis — marimo reactive notebook.

Computes pairwise Tanimoto similarity within and across toxic / non-toxic
classes and visualises the distributions as violin plots with statistical
annotation (Mann-Whitney U test).

Launch with:
    uv run marimo edit notebooks/tanimoto_analysis.py
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
    fp_type_input = mo.ui.dropdown(
        options=["morgan", "macc", "topological", "pubchem"],
        value="morgan",
        label="Fingerprint type (PubChem requires deepchem + ~1-2 s/mol network call)",
    )
    max_sample_input = mo.ui.slider(
        start=50, stop=500, step=50, value=200,
        label="Max molecules per class (limits computation time)",
    )
    mo.vstack([data_dir_input, fp_type_input, max_sample_input])
    return data_dir_input, fp_type_input, max_sample_input


@app.cell
def _load_data(data_dir_input, mo):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from src.data.loader import load_combined
    from src.data.augment import inchi_to_smiles

    try:
        X_tr_raw, y_tr_raw, X_te_raw, y_te_raw, _, _, detected_cols = (
            load_combined(data_dir_input.value)
        )

        def _conv(X, y, col):
            return inchi_to_smiles(X, y) if col == "Inchi" else (X, y)

        X_tr, y_tr = _conv(X_tr_raw, y_tr_raw, detected_cols["Train"])
        X_te, y_te = _conv(X_te_raw, y_te_raw, detected_cols["Test"])
        X_pool = X_tr + X_te
        y_pool = y_tr + y_te

        n_toxic    = sum(y_pool)
        n_nontoxic = len(y_pool) - n_toxic
        status = mo.callout(
            mo.md(f"Loaded **{len(X_pool)}** molecules — "
                  f"toxic: {n_toxic}, non-toxic: {n_nontoxic}."),
            kind="success",
        )
    except Exception as e:
        X_pool = y_pool = None
        status = mo.callout(mo.md(f"**Error loading data:** {e}"), kind="danger")

    status
    return X_pool, y_pool


@app.cell
def _compute_fps(X_pool, fp_type_input, max_sample_input, mo, np, y_pool):
    mo.stop(X_pool is None, mo.callout(mo.md("No data loaded."), kind="warn"))

    import random
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, MACCSkeys

    toxic_smi    = [s for s, y in zip(X_pool, y_pool) if y == 1]
    nontoxic_smi = [s for s, y in zip(X_pool, y_pool) if y == 0]

    n_sample = max_sample_input.value
    rng = random.Random(42)
    toxic_smi    = rng.sample(toxic_smi,    min(n_sample, len(toxic_smi)))
    nontoxic_smi = rng.sample(nontoxic_smi, min(n_sample, len(nontoxic_smi)))

    def _fps(smiles_list):
        fp_type = fp_type_input.value
        # Create PubChem featurizer once outside the per-molecule loop
        pubchem_feat = None
        if fp_type == "pubchem":
            from deepchem.feat import PubChemFingerprint
            pubchem_feat = PubChemFingerprint()
        out = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            if fp_type == "morgan":
                out.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))
            elif fp_type == "macc":
                out.append(MACCSkeys.GenMACCSKeys(mol))
            elif fp_type == "topological":
                out.append(AllChem.GetTopologicalTorsionFingerprint(mol))
            elif fp_type == "pubchem":
                canonical_smi = Chem.MolToSmiles(mol)
                try:
                    features = pubchem_feat(canonical_smi)[0]
                    if features is not None and len(features) == 881:
                        bv = DataStructs.SparseBitVect(881)
                        bv.SetBitsFromList(np.where(features == 1)[0].tolist())
                        out.append(bv)
                except Exception:
                    pass
        return out

    tox_fps = _fps(toxic_smi)
    non_fps = _fps(nontoxic_smi)

    mo.callout(
        mo.md(f"Fingerprints computed — toxic: {len(tox_fps)}, non-toxic: {len(non_fps)}."),
        kind="success",
    )
    return DataStructs, non_fps, tox_fps


@app.cell
def _compute_similarities(DataStructs, mo, non_fps, np, tox_fps):
    mo.stop(tox_fps is None, None)

    def _upper(fps):
        sims = []
        for i in range(len(fps)):
            sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:]))
        return np.array(sims, dtype=float)

    def _cross(fps_a, fps_b):
        sims = []
        for fp in fps_a:
            sims.extend(DataStructs.BulkTanimotoSimilarity(fp, fps_b))
        return np.array(sims, dtype=float)

    sim_tt = _upper(tox_fps)
    sim_nn = _upper(non_fps)
    sim_tn = _cross(tox_fps, non_fps)

    mo.callout(
        mo.md(
            f"Similarity pairs — T×T: {len(sim_tt):,}  "
            f"N×N: {len(sim_nn):,}  T×N: {len(sim_tn):,}"
        ),
        kind="success",
    )
    return sim_nn, sim_tn, sim_tt


@app.cell
def _stats_table(mo, np, sim_nn, sim_tn, sim_tt):
    mo.stop(sim_tt is None, None)

    def _run():
        import pandas as pd
        from scipy.stats import mannwhitneyu

        table_rows = []
        for lbl, va, vb in [
            ("Toxic×Toxic vs Toxic×Non-toxic",        sim_tt, sim_tn),
            ("Non-toxic×Non-toxic vs Toxic×Non-toxic", sim_nn, sim_tn),
            ("Toxic×Toxic vs Non-toxic×Non-toxic",     sim_tt, sim_nn),
        ]:
            stat, pval = mannwhitneyu(va, vb, alternative="two-sided")
            table_rows.append({
                "Comparison":  lbl,
                "Mean (A)":    round(np.mean(va), 4),
                "Mean (B)":    round(np.mean(vb), 4),
                "Median (A)":  round(np.median(va), 4),
                "Median (B)":  round(np.median(vb), 4),
                "U statistic": int(stat),
                "p-value":     f"{pval:.2e}",
            })
        return mo.ui.table(pd.DataFrame(table_rows))

    _run()
    return


@app.cell
def _violin_plot(fp_type_input, mo, np, plt, sim_nn, sim_tn, sim_tt):
    mo.stop(sim_tt is None, None)

    def _run():
        from scipy.stats import mannwhitneyu

        _, p1 = mannwhitneyu(sim_tt, sim_tn, alternative="two-sided")
        _, p2 = mannwhitneyu(sim_nn, sim_tn, alternative="two-sided")

        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

        for ax, (va, vb, lbl_a, lbl_b, pval) in zip(axes, [
            (sim_tt, sim_tn, "Toxic×Toxic",         "Toxic×Non-toxic", p1),
            (sim_nn, sim_tn, "Non-toxic×Non-toxic",  "Toxic×Non-toxic", p2),
        ]):
            parts = ax.violinplot(
                [va, vb], positions=[1, 2],
                showmedians=True, showextrema=True,
            )
            vcolors = ["#e07070", "#7070e0"]
            for pc, c in zip(parts["bodies"], vcolors):
                pc.set_facecolor(c)
                pc.set_alpha(0.6)
            for key in ("cmedians", "cmins", "cmaxes", "cbars"):
                parts[key].set_color("black")
                parts[key].set_linewidth(1.2)

            ax.axhline(np.mean(va), color=vcolors[0], linestyle="--",
                       linewidth=1, label=f"mean={np.mean(va):.3f}")
            ax.axhline(np.mean(vb), color=vcolors[1], linestyle="--",
                       linewidth=1, label=f"mean={np.mean(vb):.3f}")
            ax.set_xticks([1, 2])
            ax.set_xticklabels([lbl_a, lbl_b], rotation=8, ha="right")
            ax.set_ylabel("Tanimoto Similarity")
            ax.set_ylim(0, 1.05)
            ax.set_title(f"Mann-Whitney U  p = {pval:.2e}")
            ax.legend(fontsize=8)

        fig.suptitle(
            f"Tanimoto Similarity Distributions  "
            f"({fp_type_input.value.upper()} fingerprint)",
            fontsize=13,
        )
        plt.tight_layout()
        return fig

    _run()
    return


@app.cell
def _hist_plot(mo, np, plt, sim_nn, sim_tn, sim_tt):
    mo.stop(sim_tt is None, None)

    def _run():
        fig, ax = plt.subplots(figsize=(9, 4))
        kw = dict(bins=40, density=True, histtype="step", linewidth=1.6)
        ax.hist(sim_tt, label=f"Toxic×Toxic (μ={np.mean(sim_tt):.3f})",
                color="#e07070", **kw)
        ax.hist(sim_nn, label=f"Non-toxic×Non-toxic (μ={np.mean(sim_nn):.3f})",
                color="#70b070", **kw)
        ax.hist(sim_tn, label=f"Toxic×Non-toxic (μ={np.mean(sim_tn):.3f})",
                color="#7070e0", **kw)
        ax.set_xlabel("Tanimoto Similarity")
        ax.set_ylabel("Density")
        ax.set_title("Similarity Distribution Overlay")
        ax.legend()
        plt.tight_layout()
        return fig

    _run()
    return


if __name__ == "__main__":
    app.run()
