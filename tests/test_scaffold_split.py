"""Tests for src/data/scaffold_split.py"""

import pytest

from src.data.scaffold_split import compute_scaffold, scaffold_split


# ── Molecule pools ──────────────────────────────────────────────────────────

# Substituted benzenes — all share the benzene Bemis-Murcko scaffold
BENZENE_POOL = [
    "Cc1ccccc1",             # toluene
    "Nc1ccccc1",             # aniline
    "Oc1ccccc1",             # phenol
    "OC(=O)c1ccccc1",       # benzoic acid
    "Clc1ccccc1",            # chlorobenzene
    "Fc1ccccc1",             # fluorobenzene
    "Brc1ccccc1",            # bromobenzene
    "[O-][N+](=O)c1ccccc1",  # nitrobenzene
]

# Naphthalene compounds — all share the naphthalene scaffold
NAPHTHALENE_POOL = [
    "c1ccc2ccccc2c1",        # naphthalene
    "Cc1ccc2ccccc2c1",       # 2-methylnaphthalene
    "Oc1ccc2ccccc2c1",       # 2-naphthol
    "Nc1ccc2ccccc2c1",       # 2-naphthylamine
]

# Acyclic molecules — Bemis-Murcko scaffold is empty ("")
ACYCLIC_POOL = [
    "CCO",                   # ethanol
    "CC(=O)O",              # acetic acid
    "CCCC",                 # butane
    "CN",                    # methylamine
    "CCC",                   # propane
]


def _make_pool():
    """17-molecule dataset: 8 benzene + 4 naphthalene + 5 acyclic,
    alternating 0/1 labels."""
    smiles = BENZENE_POOL + NAPHTHALENE_POOL + ACYCLIC_POOL
    labels = [i % 2 for i in range(len(smiles))]
    return smiles, labels


# ── compute_scaffold ────────────────────────────────────────────────────────

class TestComputeScaffold:

    def test_benzene_derivatives_all_share_same_scaffold(self):
        scaffolds = {compute_scaffold(s) for s in BENZENE_POOL}
        assert len(scaffolds) == 1
        assert "" not in scaffolds

    def test_naphthalene_derivatives_all_share_same_scaffold(self):
        scaffolds = {compute_scaffold(s) for s in NAPHTHALENE_POOL}
        assert len(scaffolds) == 1
        assert "" not in scaffolds

    def test_benzene_and_naphthalene_scaffolds_are_different(self):
        sc_benz = compute_scaffold(BENZENE_POOL[0])
        sc_naph = compute_scaffold(NAPHTHALENE_POOL[0])
        assert sc_benz != sc_naph

    def test_side_chains_are_stripped(self):
        # toluene, aniline, and bare benzene all reduce to the same scaffold
        sc_toluene = compute_scaffold("Cc1ccccc1")
        sc_aniline  = compute_scaffold("Nc1ccccc1")
        sc_benzene  = compute_scaffold("c1ccccc1")
        assert sc_toluene == sc_aniline == sc_benzene

    def test_acyclic_molecules_return_empty_string(self):
        for smi in ACYCLIC_POOL:
            assert compute_scaffold(smi) == "", f"Expected '' for acyclic SMILES {smi!r}"

    def test_invalid_smiles_return_empty_string(self):
        for bad in ["not_a_smiles", "C(C(C", "", "!!!"]:
            assert compute_scaffold(bad) == "", f"Expected '' for invalid SMILES {bad!r}"


# ── scaffold_split ──────────────────────────────────────────────────────────

class TestScaffoldSplit:

    def test_partition_covers_full_dataset(self):
        smiles, labels = _make_pool()
        X_tr, y_tr, X_te, y_te = scaffold_split(smiles, labels, test_frac=0.2, seed=42)
        assert len(X_tr) + len(X_te) == len(smiles)
        assert len(y_tr) + len(y_te) == len(labels)

    def test_no_molecule_appears_in_both_partitions(self):
        smiles, labels = _make_pool()
        X_tr, _, X_te, _ = scaffold_split(smiles, labels, test_frac=0.2, seed=42)
        assert set(X_tr).isdisjoint(set(X_te))

    def test_no_scaffold_overlap_between_partitions(self):
        smiles, labels = _make_pool()
        X_tr, _, X_te, _ = scaffold_split(smiles, labels, test_frac=0.2, seed=42)
        train_scaffolds = {compute_scaffold(s) for s in X_tr}
        test_scaffolds  = {compute_scaffold(s) for s in X_te}
        overlap = train_scaffolds & test_scaffolds
        assert overlap == set(), f"Scaffold overlap detected: {overlap}"

    def test_molecules_of_same_scaffold_stay_together(self):
        smiles, labels = _make_pool()
        X_tr, _, X_te, _ = scaffold_split(smiles, labels, test_frac=0.2, seed=42)
        tr_set, te_set = set(X_tr), set(X_te)
        for pool_name, pool in [
            ("benzene",     BENZENE_POOL),
            ("naphthalene", NAPHTHALENE_POOL),
            ("acyclic",     ACYCLIC_POOL),
        ]:
            in_train = sum(s in tr_set for s in pool)
            in_test  = sum(s in te_set for s in pool)
            assert in_train == 0 or in_test == 0, (
                f"Scaffold group '{pool_name}' was split across partitions: "
                f"{in_train} in train, {in_test} in test"
            )

    def test_labels_are_aligned_with_smiles(self):
        smiles, labels = _make_pool()
        original = dict(zip(smiles, labels))
        X_tr, y_tr, X_te, y_te = scaffold_split(smiles, labels, test_frac=0.2, seed=42)
        for smi, lab in zip(X_tr, y_tr):
            assert original[smi] == lab, f"Label mismatch in train for {smi!r}"
        for smi, lab in zip(X_te, y_te):
            assert original[smi] == lab, f"Label mismatch in test for {smi!r}"

    def test_test_size_within_one_group_of_target(self):
        # The sort-descending / fill-train-first algorithm:
        # train fills up first, so test may be slightly less than the target
        # (if the last group that fills train overshoots).
        # Actual deviation is bounded by the size of the largest scaffold group.
        smiles, labels = _make_pool()
        n = len(smiles)
        test_frac = 0.2
        _, _, X_te, _ = scaffold_split(smiles, labels, test_frac=test_frac, seed=42)
        n_train_target = int(n * (1 - test_frac))
        n_test_expected = n - n_train_target
        max_group_size = max(len(BENZENE_POOL), len(NAPHTHALENE_POOL), len(ACYCLIC_POOL))
        assert abs(len(X_te) - n_test_expected) <= max_group_size

    def test_split_is_deterministic_regardless_of_seed(self):
        # The sort-based algorithm is fully deterministic; seed is ignored.
        smiles, labels = _make_pool()
        r0  = scaffold_split(smiles, labels, test_frac=0.2, seed=0)
        r42 = scaffold_split(smiles, labels, test_frac=0.2, seed=42)
        r99 = scaffold_split(smiles, labels, test_frac=0.2, seed=99)
        assert r0[0] == r42[0] == r99[0], "X_train differs across seeds (should be deterministic)"
        assert r0[2] == r42[2] == r99[2], "X_test differs across seeds (should be deterministic)"

    def test_mismatched_smiles_and_labels_raises(self):
        with pytest.raises(AssertionError):
            scaffold_split(["CCO", "c1ccccc1"], [0], test_frac=0.2, seed=42)

    def test_all_acyclic_pool_lands_in_one_partition(self):
        """No ring systems → single "" scaffold group → cannot be split."""
        smiles = list(ACYCLIC_POOL)
        labels = [0] * len(smiles)
        X_tr, _, X_te, _ = scaffold_split(smiles, labels, test_frac=0.5, seed=42)
        assert len(X_tr) == 0 or len(X_te) == 0, (
            "All-acyclic pool must land entirely in one partition"
        )
        assert len(X_tr) + len(X_te) == len(smiles)

    def test_single_scaffold_group_lands_in_one_partition(self):
        """One scaffold group → all molecules go to test (greedy fills test first)."""
        smiles = list(BENZENE_POOL)
        labels = [0] * len(smiles)
        X_tr, _, X_te, _ = scaffold_split(smiles, labels, test_frac=0.3, seed=42)
        assert len(X_tr) == 0 or len(X_te) == 0, (
            "Single scaffold group must land entirely in one partition"
        )
        assert len(X_tr) + len(X_te) == len(smiles)

    def test_zero_test_fraction_puts_all_in_train(self):
        smiles, labels = _make_pool()
        X_tr, y_tr, X_te, y_te = scaffold_split(smiles, labels, test_frac=0.0, seed=42)
        assert len(X_te) == 0
        assert len(X_tr) == len(smiles)

    def test_split_invariants_hold_across_fractions(self):
        """Completeness and no-overlap hold for a range of test fractions."""
        smiles, labels = _make_pool()
        for frac in (0.1, 0.2, 0.3, 0.5):
            X_tr, _, X_te, _ = scaffold_split(smiles, labels, test_frac=frac)
            assert len(X_tr) + len(X_te) == len(smiles), f"frac={frac}: incomplete partition"
            assert set(X_tr).isdisjoint(set(X_te)), f"frac={frac}: partition overlap"
            sc_tr = {compute_scaffold(s) for s in X_tr}
            sc_te = {compute_scaffold(s) for s in X_te}
            assert sc_tr.isdisjoint(sc_te), f"frac={frac}: scaffold overlap"
