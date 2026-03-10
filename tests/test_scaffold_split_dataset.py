"""
Integration test: scaffold split on the real Train_inchi + Test dataset.

Validates the split and prints a table of scaffold groups with per-group
class-balance stats.  The full table is also written to
output/scaffold_groups.csv so it can be opened in Excel.

Run with:
    uv run pytest tests/test_scaffold_split_dataset.py -v -s
The -s flag keeps stdout visible so the printed table appears in the terminal.
"""

import os
from collections import defaultdict

import pandas as pd
import pytest

from src.data.scaffold_split import compute_scaffold, scaffold_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR    = os.path.join(_ROOT, "input")
TRAIN_FILE  = os.path.join(DATA_DIR, "Train_inchi.xlsx")
TEST_FILE   = os.path.join(DATA_DIR, "Test.xlsx")
OUTPUT_CSV  = os.path.join(_ROOT, "output", "scaffold_groups.csv")

_data_present = os.path.exists(TRAIN_FILE) and os.path.exists(TEST_FILE)
needs_data = pytest.mark.skipif(
    not _data_present,
    reason="Real dataset not found in input/ — skipping integration tests",
)

# ---------------------------------------------------------------------------
# Module-scoped fixtures (load & split once for all tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pool():
    """Load Train_inchi + Test.xlsx, convert to SMILES, return combined pool."""
    from src.data.augment import inchi_to_smiles
    from src.data.loader import load_combined

    X_tr_raw, y_tr_raw, X_te_raw, y_te_raw, _, _, detected_cols = load_combined(DATA_DIR)

    def _conv(X, y, col, name):
        if col == "Inchi":
            return inchi_to_smiles(X, y)
        return X, y

    X_tr, y_tr = _conv(X_tr_raw, y_tr_raw, detected_cols["Train"], "Train")
    X_te, y_te = _conv(X_te_raw, y_te_raw, detected_cols["Test"],  "Test")

    return X_tr + X_te, y_tr + y_te


@pytest.fixture(scope="module")
def split(pool):
    X_pool, y_pool = pool
    # test_frac=0.12 matches the original Train_inchi (7357) / Test (1000) file ratio
    X_tr, y_tr, X_te, y_te = scaffold_split(X_pool, y_pool, test_frac=0.12)
    return X_tr, y_tr, X_te, y_te


# ---------------------------------------------------------------------------
# Invariant tests
# ---------------------------------------------------------------------------

@needs_data
def test_split_covers_all_molecules(pool, split):
    X_pool, y_pool = pool
    X_tr, y_tr, X_te, y_te = split
    assert len(X_tr) + len(X_te) == len(X_pool), (
        f"Molecules lost: pool={len(X_pool)}, "
        f"train={len(X_tr)}, test={len(X_te)}"
    )
    assert len(y_tr) + len(y_te) == len(y_pool)


@needs_data
def test_no_scaffold_overlap(split):
    X_tr, _, X_te, _ = split
    sc_tr = {compute_scaffold(s) for s in X_tr}
    sc_te = {compute_scaffold(s) for s in X_te}
    overlap = sc_tr & sc_te
    assert overlap == set(), (
        f"{len(overlap)} scaffold(s) appear in both train and test:\n"
        + "\n".join(f"  {sc!r}" for sc in list(overlap)[:5])
    )


@needs_data
def test_no_molecule_duplicated_across_partitions(split):
    X_tr, _, X_te, _ = split
    shared = set(X_tr) & set(X_te)
    assert shared == set(), (
        f"{len(shared)} SMILES appear in both train and test"
    )


# ---------------------------------------------------------------------------
# Reporting test
# ---------------------------------------------------------------------------

@needs_data
def test_scaffold_groups_report(pool, split):
    """
    Build per-scaffold stats, print a summary table to stdout, and save the
    full table to output/scaffold_groups.csv.

    Assertions at the end ensure the table is internally consistent.
    """
    X_pool, y_pool = pool
    X_tr, y_tr, X_te, y_te = split

    # ── Build scaffold → partition map (each scaffold is entirely in one side)
    sc_partition: dict[str, str] = {}
    for smi in X_tr:
        sc_partition[compute_scaffold(smi)] = "Train"
    for smi in X_te:
        sc_partition[compute_scaffold(smi)] = "Test"

    # ── Accumulate per-scaffold stats from the pool
    stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "toxic": 0})
    for smi, lab in zip(X_pool, y_pool):
        sc = compute_scaffold(smi)
        stats[sc]["total"] += 1
        stats[sc]["toxic"] += lab

    # ── Build DataFrame
    rows = []
    for sc, s in stats.items():
        total     = s["total"]
        toxic     = s["toxic"]
        non_toxic = total - toxic
        rows.append({
            "scaffold":  sc if sc else "(acyclic)",
            "total":     total,
            "toxic":     toxic,
            "non_toxic": non_toxic,
            "pct_toxic": round(100.0 * toxic / total, 1) if total else 0.0,
            "partition": sc_partition.get(sc, "unknown"),
        })

    df = (
        pd.DataFrame(rows)
        .sort_values("total", ascending=False)
        .reset_index(drop=True)
    )
    df.index += 1  # 1-based rank

    # ── Summary stats
    n_total    = len(X_pool)
    n_train    = len(X_tr)
    n_test     = len(X_te)
    n_groups   = len(df)
    n_tr_grp   = int((df["partition"] == "Train").sum())
    n_te_grp   = int((df["partition"] == "Test").sum())
    tr_tox_pct = 100.0 * sum(y_tr) / len(y_tr) if y_tr else 0.0
    te_tox_pct = 100.0 * sum(y_te) / len(y_te) if y_te else 0.0

    # ── Print summary
    W = 74
    print()
    print("=" * W)
    print("  Scaffold Split — Real Dataset Report")
    print("=" * W)
    print(f"  Molecules  : {n_total:>5}  "
          f"(train {n_train} / test {n_test})")
    print(f"  Test frac  : {n_test / n_total:.3f}  (target 0.200)")
    print(f"  Scaffolds  : {n_groups:>5}  "
          f"(train {n_tr_grp} groups / test {n_te_grp} groups)")
    print(f"  % Toxic    :         "
          f"train {tr_tox_pct:.1f}%  /  test {te_tox_pct:.1f}%")
    print("=" * W)

    # ── Print top-N table
    TOP_N = 30
    print(f"\n  Top {TOP_N} scaffold groups (by size):\n")
    hdr = (
        f"  {'Rank':>4}  {'Total':>5}  "
        f"{'Toxic':>5}  {'Non-tox':>7}  {'%Toxic':>6}  "
        f"{'Part':>5}  Scaffold"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for rank, row in df.head(TOP_N).iterrows():
        sc_disp = row["scaffold"]
        if len(sc_disp) > 38:
            sc_disp = sc_disp[:35] + "..."
        print(
            f"  {rank:>4}  {row['total']:>5}  "
            f"{row['toxic']:>5}  {row['non_toxic']:>7}  "
            f"{row['pct_toxic']:>5.1f}%  "
            f"{row['partition']:>5}  {sc_disp}"
        )

    remaining = n_groups - TOP_N
    if remaining > 0:
        print(f"\n  ... {remaining} more group(s) in the saved CSV.")

    # ── Save full table (non-fatal if the file is locked, e.g. open in Excel)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    try:
        df.to_csv(OUTPUT_CSV, index_label="rank")
        print(f"\n  Full table ({n_groups} rows) saved → {OUTPUT_CSV}")
    except PermissionError:
        print(f"\n  WARNING: could not write {OUTPUT_CSV} (file locked — close it in Excel and re-run)")
    print("=" * W + "\n")

    # ── Assertions
    assert (df["partition"] == "Train").any(),  "No scaffolds assigned to train"
    assert (df["partition"] == "Test").any(),   "No scaffolds assigned to test"
    assert df["total"].sum() == n_total,        "Group sizes don't sum to total pool size"
    assert df["toxic"].sum() == sum(y_pool),    "Toxic counts don't sum correctly"
    assert (df["non_toxic"] >= 0).all(),        "Negative non-toxic count"
    # Each scaffold group must be entirely in one partition (not "unknown")
    assert (df["partition"] != "unknown").all(), "Some scaffold groups have no partition"
