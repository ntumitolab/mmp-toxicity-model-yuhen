"""
Structural-alert & substructure analysis — marimo reactive notebook.

Two complementary analyses:
1. PAINS / Brenk alerts  — RDKit FilterCatalog flags well-known
   problematic substructures; shows prevalence in toxic vs non-toxic.
2. Ring-substructure frequency — finds ring fragments enriched in the
   toxic class (reproducing the approach of the original notebook).

Launch with:
    uv run marimo edit notebooks/structure_alerts.py
"""

import marimo

__generated_with = "0.20.2"
app = marimo.App()


@app.cell
def _imports():
    import marimo as mo
    import matplotlib.pyplot as plt

    return mo, plt


@app.cell
def _controls(mo):
    data_dir_input = mo.ui.text(value="input", label="Data directory")
    min_freq_input = mo.ui.slider(
        start=2, stop=30, step=1, value=5,
        label="Min frequency to report a ring substructure",
    )
    top_n_input = mo.ui.slider(
        start=5, stop=30, step=5, value=5,
        label="Top-N substructures to display",
    )
    mo.vstack([data_dir_input, min_freq_input, top_n_input])
    return data_dir_input, min_freq_input, top_n_input


@app.cell
def _load_data(data_dir_input, mo):
    def _load():
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

        from src.data.loader import load_combined
        from src.data.augment import inchi_to_smiles
        from rdkit import Chem

        X_tr_raw, y_tr_raw, X_te_raw, y_te_raw, _, _, detected_cols = (
            load_combined(data_dir_input.value)
        )

        def _conv(X, y, col):
            return inchi_to_smiles(X, y) if col == "Inchi" else (X, y)

        X_tr, y_tr = _conv(X_tr_raw, y_tr_raw, detected_cols["Train"])
        X_te, y_te = _conv(X_te_raw, y_te_raw, detected_cols["Test"])
        X_pool = X_tr + X_te
        y_pool = y_tr + y_te

        mols_out, labels_out = [], []
        for smi, lab in zip(X_pool, y_pool):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                mols_out.append(mol)
                labels_out.append(lab)

        toxic_out    = [m for m, y in zip(mols_out, labels_out) if y == 1]
        nontoxic_out = [m for m, y in zip(mols_out, labels_out) if y == 0]
        return toxic_out, nontoxic_out, mols_out, labels_out

    try:
        toxic_mols, nontoxic_mols, mols, labels_m = _load()
        status = mo.callout(
            mo.md(
                f"Loaded **{len(mols)}** valid molecules — "
                f"toxic: {len(toxic_mols)}, non-toxic: {len(nontoxic_mols)}."
            ),
            kind="success",
        )
    except Exception as e:
        toxic_mols = nontoxic_mols = mols = labels_m = None
        status = mo.callout(mo.md(f"**Error:** {e}"), kind="danger")

    status
    return nontoxic_mols, toxic_mols


@app.cell
def _pains_analysis(mo, nontoxic_mols, toxic_mols):
    mo.stop(toxic_mols is None, None)

    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    catalog = FilterCatalog(params)

    def _flag(mol_list):
        flagged, alert_names = [], []
        for mol in mol_list:
            entry = catalog.GetFirstMatch(mol)
            if entry:
                flagged.append(True)
                alert_names.append(entry.GetDescription())
            else:
                flagged.append(False)
                alert_names.append(None)
        return flagged, alert_names

    tox_flags,  tox_alerts  = _flag(toxic_mols)
    non_flags,  non_alerts  = _flag(nontoxic_mols)

    n_tox_hit = sum(tox_flags)
    n_non_hit = sum(non_flags)
    pct_tox   = 100 * n_tox_hit / len(toxic_mols)
    pct_non   = 100 * n_non_hit / len(nontoxic_mols)

    mo.callout(
        mo.md(
            f"PAINS/Brenk alerts — "
            f"toxic: **{n_tox_hit}/{len(toxic_mols)} ({pct_tox:.1f}%)**  "
            f"non-toxic: **{n_non_hit}/{len(nontoxic_mols)} ({pct_non:.1f}%)**"
        ),
        kind="info",
    )
    # Return separate variables so downstream cells receive them by name
    import pandas as pd
    pains_summary = pd.DataFrame({
        "Class":           ["Toxic",        "Non-toxic"],
        "Total":           [len(toxic_mols), len(nontoxic_mols)],
        "PAINS/Brenk hit": [n_tox_hit,      n_non_hit],
        "Hit rate (%)":    [round(pct_tox, 1), round(pct_non, 1)],
    })
    return non_alerts, pains_summary, pd, tox_alerts


@app.cell
def _pains_table(mo, pains_summary):
    mo.stop(pains_summary is None, None)

    def _run():
        import pandas as pd
        return mo.ui.table(pains_summary)

    _run()
    return


@app.cell
def _pains_bar(mo, pains_summary, plt):
    mo.stop(pains_summary is None, None)

    def _run():
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(
            pains_summary["Class"], pains_summary["Hit rate (%)"],
            color=["#e05050", "#4a90d9"], alpha=0.8, width=0.5,
        )
        for bar, val in zip(bars, pains_summary["Hit rate (%)"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11,
            )
        ax.set_ylabel("PAINS / Brenk alert rate (%)")
        ax.set_title("Structural Alert Prevalence by Class")
        ax.set_ylim(0, max(pains_summary["Hit rate (%)"]) * 1.25)
        plt.tight_layout()
        return fig

    _run()
    return


@app.cell
def _top_pains_alerts(mo, non_alerts, tox_alerts):
    mo.stop(tox_alerts is None, None)

    def _run():
        import pandas as pd
        from collections import Counter

        tox_counts = Counter(a for a in tox_alerts if a is not None)
        non_counts = Counter(a for a in non_alerts if a is not None)
        all_alerts = set(tox_counts) | set(non_counts)

        n_tox_alerted = max(len([x for x in tox_alerts if x is not None]), 1)
        n_non_alerted = max(len([x for x in non_alerts if x is not None]), 1)

        table_rows = sorted(
            [
                {
                    "Alert":         a,
                    "Toxic hits":    tox_counts.get(a, 0),
                    "Non-tox hits":  non_counts.get(a, 0),
                    "Enrichment (T/N)": round(
                        (tox_counts.get(a, 0) / n_tox_alerted) /
                        max(non_counts.get(a, 0) / n_non_alerted, 1e-9),
                        2,
                    ),
                }
                for a in all_alerts
            ],
            key=lambda r: r["Toxic hits"],
            reverse=True,
        )
        return mo.ui.table(pd.DataFrame(table_rows))

    _run()
    return


@app.cell
def _ring_freq(min_freq_input, mo, nontoxic_mols, pd, toxic_mols):
    mo.stop(toxic_mols is None, None)

    def _compute():
        from collections import defaultdict
        from rdkit import Chem

        def _ring_substructs(mol, min_size=3, max_size=8):
            out = []
            ssr = Chem.GetSymmSSSR(mol)
            for ring in ssr:
                ring = list(ring)
                if min_size <= len(ring) <= max_size:
                    smi = Chem.MolFragmentToSmiles(mol, ring, canonical=True)
                    if smi:
                        out.append(smi)
            return out

        def _count(mol_list):
            freq = defaultdict(int)
            for mol in mol_list:
                seen = set()
                for s in _ring_substructs(mol):
                    if s not in seen:
                        freq[s] += 1
                        seen.add(s)
            return freq

        tox_freq = _count(toxic_mols)
        non_freq = _count(nontoxic_mols)

        min_f = min_freq_input.value
        n_tox = len(toxic_mols)
        n_non = len(nontoxic_mols)

        result_rows = []
        for ring_smi in set(tox_freq) | set(non_freq):
            tf = tox_freq.get(ring_smi, 0)
            nf = non_freq.get(ring_smi, 0)
            if tf < min_f and nf < min_f:
                continue
            tox_rate = tf / n_tox
            non_rate = nf / n_non
            result_rows.append({
                "Ring SMILES":    ring_smi,
                "Toxic count":    tf,
                "Non-tox count":  nf,
                "Toxic rate":     round(tox_rate, 4),
                "Non-tox rate":   round(non_rate, 4),
                "Enrichment T/N": round(tox_rate / max(non_rate, 1e-9), 3),
            })
        return result_rows

    _rows = _compute()
    df_rings = (
        pd.DataFrame(_rows)
        .sort_values("Enrichment T/N", ascending=False)
        .reset_index(drop=True)
    )
    df_rings.index += 1

    mo.callout(
        mo.md(
            f"Found **{len(df_rings)}** ring substructures "
            f"appearing ≥ {min_freq_input.value}× in at least one class."
        ),
        kind="info",
    )
    return (df_rings,)


@app.cell
def _ring_table(df_rings, mo, top_n_input):
    mo.stop(df_rings is None, None)

    def _run():
        return mo.ui.table(df_rings.head(top_n_input.value))

    _run()
    return


@app.cell
def _ring_bar(df_rings, mo, plt, top_n_input):
    mo.stop(df_rings is None, None)

    def _run():
        import numpy as np

        top = df_rings.head(top_n_input.value)
        x = np.arange(len(top))
        w = 0.38

        fig, ax = plt.subplots(figsize=(max(10, len(top) * 0.7), 5))
        ax.bar(x - w / 2, top["Toxic rate"],   w, label="Toxic",     color="#e05050", alpha=0.8)
        ax.bar(x + w / 2, top["Non-tox rate"], w, label="Non-toxic", color="#4a90d9", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(top["Ring SMILES"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Fraction of class containing ring")
        ax.set_title(f"Top-{len(top)} Toxic-enriched Ring Substructures")
        ax.legend()
        plt.tight_layout()
        return fig

    _run()
    return


@app.cell
def _ring_images(df_rings, mo, plt, top_n_input):
    mo.stop(df_rings is None, None)

    def _run():
        import math
        from rdkit import Chem
        from rdkit.Chem import Draw

        smiles_list = df_rings["Ring SMILES"].head(top_n_input.value).tolist()
        ring_mols = [Chem.MolFromSmiles(s) for s in smiles_list if Chem.MolFromSmiles(s)]
        leg = [
            f"{row['Ring SMILES']}\nT:{row['Toxic count']} N:{row['Non-tox count']} ×{row['Enrichment T/N']:.1f}"
            for _, row in df_rings.head(len(ring_mols)).iterrows()
        ]

        n_cols = min(5, len(ring_mols))
        n_rows = math.ceil(len(ring_mols) / n_cols)
        img = Draw.MolsToGridImage(
            ring_mols, molsPerRow=n_cols,
            subImgSize=(250, 200),
            legends=leg,
            returnPNG=False,
        )

        fig, ax = plt.subplots(figsize=(n_cols * 2.8, n_rows * 2.4))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title("Top Toxic-enriched Ring Substructures", pad=10)
        plt.tight_layout()
        return fig

    _run()
    return


if __name__ == "__main__":
    app.run()
