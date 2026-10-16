"""Site views and site tables: validation never raises; pseudobulk and bulk transforms are exact.

Offline and fast: a hand-computable toy h5ad and a toy bulk matrix written to tmp_path.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.stats import norm

from agentcoop.coop.claims import Vocabulary
from agentcoop.coop.view import load_view, site_table, write_view

VOCAB = Vocabulary(populations=("CD4 T", "B", "NK"), groups=("control", "treated", "mimic"),
                   contrasts=(("treated", "control"), ("mimic", "control")))

FEATURES = ["ENSG1", "ENSG2", "ENSG3", "ENSG4"]
#: ENSG2 and ENSG3 share a symbol (summed); ENSG4 is unmapped (dropped from the genes but still
#: in the library size); ENSG9 is mapped but absent from the data (ignored).
GENES = {"ENSG1": "gzmb", "ENSG2": "CD3E", "ENSG3": " cd3e ", "ENSG4": "", "ENSG9": "MS4A1"}
#: cell -> (sample, population, counts over FEATURES)
CELLS = {
    "c1": ("s1", "CD4 T", [1, 2, 3, 4]),
    "c2": ("s1", "CD4 T", [0, 1, 1, 2]),
    "c3": ("s1", "B", [5, 0, 0, 5]),
    "c4": ("s1", "other", [9, 9, 9, 9]),      # outside the vocabulary: never in a unit
    "c5": ("s2", "CD4 T", [2, 2, 2, 2]),
    "c6": ("s2", "B", [1, 0, 3, 0]),
    "c7": ("s2", "B", [0, 4, 0, 4]),
    "c8": ("s3", "CD4 T", [7, 7, 7, 7]),      # sample s3 is excluded
}
UNLISTED = ("c9", [3, 3, 3, 3])               # in the data, not in cells.tsv
SAMPLES = pd.DataFrame({"sample": ["s1", "s2", "s3"], "donor": ["d1", "d2", "d3"],
                        "group": ["control", "treated", "exclude"],
                        "assay": ["10x v3", "10x v3", "10x v2"]})


def _cells_frame() -> pd.DataFrame:
    return pd.DataFrame([(c, s, p) for c, (s, p, _) in CELLS.items()],
                        columns=["cell", "sample", "population"])


def _scrna(tmp_path: Path, **view) -> tuple[Path, Path]:
    """Toy h5ad (sparse raw counts) plus a valid view; keyword arguments replace view parts."""
    x = np.array([v[2] for v in CELLS.values()] + [UNLISTED[1]], dtype=np.float32)
    tmp_path.mkdir(parents=True, exist_ok=True)
    data = tmp_path / "site.h5ad"
    ad.AnnData(X=sp.csr_matrix(x), obs=pd.DataFrame(index=[*CELLS, UNLISTED[0]]),
               var=pd.DataFrame(index=FEATURES)).write_h5ad(data)
    parts = {"genes": GENES, "samples": SAMPLES.copy(), "cells": _cells_frame(),
             "spec": {"normalization": "cpm", "unit": "donor"}}
    parts.update(view)
    root = tmp_path / "view"
    write_view(root, **parts)
    return root, data


def test_valid_view_normalizes_symbols_and_reports_dropped_rows(tmp_path):
    root, _ = _scrna(tmp_path)
    view, warnings = load_view(root, kind="scrna", vocab=VOCAB)
    assert view is not None
    assert view.genes == {"ENSG1": "GZMB", "ENSG2": "CD3E", "ENSG3": "CD3E", "ENSG9": "MS4A1"}
    assert warnings == ["genes.tsv: 1 row(s) without a symbol dropped"]
    assert list(view.samples.columns) == ["sample", "donor", "group", "assay"]
    assert view.cells["cell"].tolist() == list(CELLS)
    assert view.spec == {"normalization": "cpm", "unit": "donor"}


def _edit(path: Path, change) -> None:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    change(df).to_csv(path, sep="\t", index=False)


MALFORMED = {
    "missing column": (lambda r: _edit(r / "samples.tsv", lambda d: d.drop(columns="group")),
                       "samples.tsv: missing column(s) group"),
    "unknown group": (lambda r: _edit(r / "samples.tsv",
                                      lambda d: d.replace({"group": {"control": "ctrl"}})),
                      "group(s) 'ctrl' not in the vocabulary"),
    "no vocabulary group": (lambda r: _edit(r / "samples.tsv", lambda d: d.assign(group="exclude")),
                            "no sample in a vocabulary group"),
    "duplicate sample": (lambda r: _edit(r / "samples.tsv",
                                         lambda d: d.replace({"sample": {"s2": "s1"}})),
                         "duplicate sample 's1'"),
    "empty donor": (lambda r: _edit(r / "samples.tsv", lambda d: d.assign(donor="")),
                    "3 row(s) with an empty donor"),
    "duplicate cell id": (lambda r: _edit(r / "cells.tsv",
                                          lambda d: d.replace({"cell": {"c2": "c1"}})),
                          "duplicate cell 'c1'"),
    "unknown population": (lambda r: _edit(r / "cells.tsv",
                                           lambda d: d.replace({"population": {"B": "B cells"}})),
                           "population(s) 'B cells' not in the vocabulary"),
    "unknown sample": (lambda r: _edit(r / "cells.tsv",
                                       lambda d: d.replace({"sample": {"s2": "s7"}})),
                       "3 cell(s) reference unknown sample(s) 's7'"),
    "duplicate gene id": (lambda r: _edit(r / "genes.tsv",
                                          lambda d: d.replace({"native_id": {"ENSG2": "ENSG1"}})),
                          "duplicate native_id 'ENSG1'"),
    "no mapped gene": (lambda r: _edit(r / "genes.tsv", lambda d: d.assign(symbol="")),
                       "no feature mapped to a symbol"),
    "missing cells.tsv": (lambda r: (r / "cells.tsv").unlink(), "cells.tsv: missing"),
    "empty samples.tsv": (lambda r: (r / "samples.tsv").write_text(""), "samples.tsv: empty"),
    "duplicate column": (lambda r: (r / "samples.tsv").write_text(
                             "sample\tdonor\tgroup\tgroup\ns1\td1\tcontrol\ttreated\n"),
                         "samples.tsv: duplicate column(s) 'group'"),
    "binary genes.tsv": (lambda r: (r / "genes.tsv").write_bytes(bytes(range(256)) * 4),
                         "genes.tsv: unreadable"),
    "ragged cells.tsv": (lambda r: (r / "cells.tsv").write_text(
                             "cell\tsample\tpopulation\nc1\ts1\tB\textra\tmore\n"),
                         "cells.tsv: 1 row(s) whose field count differs from the header"),
    "spec not an object": (lambda r: (r / "spec.json").write_text("[1, 2]"),
                           "spec.json: must be a JSON object"),
    "spec malformed": (lambda r: (r / "spec.json").write_text("{"), "spec.json: unreadable"),
    "spec missing": (lambda r: (r / "spec.json").unlink(), "spec.json: missing"),
}


@pytest.mark.parametrize("case", sorted(MALFORMED))
def test_malformed_view_returns_errors_never_raises(tmp_path, case):
    mutate, expected = MALFORMED[case]
    root, _ = _scrna(tmp_path)
    mutate(root)
    view, errors = load_view(root, kind="scrna", vocab=VOCAB)
    assert view is None
    assert any(expected in e for e in errors), errors


def test_every_defect_is_reported_at_once(tmp_path):
    root, _ = _scrna(tmp_path)
    MALFORMED["unknown group"][0](root)
    MALFORMED["duplicate cell id"][0](root)
    MALFORMED["spec malformed"][0](root)
    view, errors = load_view(root, kind="scrna", vocab=VOCAB)
    assert view is None and len(errors) == 3


def test_missing_view_directory_is_an_error(tmp_path):
    assert load_view(tmp_path / "nope", kind="scrna", vocab=VOCAB) == (
        None, [f"view directory {tmp_path / 'nope'} missing"])


def _scrna_table(tmp_path: Path, **view):
    root, data = _scrna(tmp_path, **view)
    loaded, problems = load_view(root, kind="scrna", vocab=VOCAB)
    assert loaded is not None, problems
    return site_table(loaded, data, kind="scrna")


def test_pseudobulk_sums_symbols_and_cpm(tmp_path):
    table = _scrna_table(tmp_path)
    assert table.kind == "scrna"
    assert table.genes == ["CD3E", "GZMB"]
    assert table.units.to_dict("list") == {
        "unit": ["s1|B", "s1|CD4 T", "s2|B", "s2|CD4 T"], "donor": ["d1", "d1", "d2", "d2"],
        "group": ["control", "control", "treated", "treated"],
        "population": ["B", "CD4 T", "B", "CD4 T"], "n_cells": [1, 2, 2, 1]}
    # per unit: CD3E = ENSG2 + ENSG3, GZMB = ENSG1; library = all features, unmapped ENSG4 included
    sums = np.array([[0, 5], [7, 1], [7, 1], [4, 2]], dtype=float)
    library = np.array([10, 14, 12, 8], dtype=float)
    cpm = sums / library[:, None] * 1e6
    np.testing.assert_allclose(table.values, np.log2(cpm + 1))
    np.testing.assert_allclose(table.population_mean_cpm("B"), cpm[[0, 2]].mean(axis=0))
    np.testing.assert_allclose(table.population_mean_cpm("CD4 T"), cpm[[1, 3]].mean(axis=0))


def test_other_excluded_and_unlisted_cells_never_enter_a_unit(tmp_path):
    table = _scrna_table(tmp_path)
    assert "other" not in set(table.units["population"])
    assert "exclude" not in set(table.units["group"])
    assert table.units["n_cells"].sum() == 6            # c1-c3, c5-c7; not c4, c8, c9
    # a population the site lacks has no measurable gene (tester -> untestable, never supported)
    np.testing.assert_array_equal(table.population_mean_cpm("NK"), [0.0, 0.0])


def test_composition_fractions_of_kept_cells(tmp_path):
    comp = _scrna_table(tmp_path).composition
    assert list(comp.index) == ["s1", "s2"]
    assert list(comp.columns) == ["B", "CD4 T", "group", "donor"]
    np.testing.assert_allclose(comp[["B", "CD4 T"]].to_numpy(dtype=float),
                               [[1 / 3, 2 / 3], [2 / 3, 1 / 3]])  # c4 ("other") not counted
    assert comp["group"].tolist() == ["control", "treated"]
    assert comp["donor"].tolist() == ["d1", "d2"]


def test_pseudobulk_matches_a_naive_loop_on_dense_counts(tmp_path):
    rng = np.random.default_rng(0)
    n_cells, n_feat = 400, 30
    x = rng.poisson(2.0, size=(n_cells, n_feat)).astype(np.int64)
    feats = [f"F{j}" for j in range(n_feat)]
    genes = {f: f"G{j % 20}" for j, f in enumerate(feats)}          # F20..F29 duplicate G0..G9
    samples = pd.DataFrame({"sample": [f"s{i}" for i in range(6)],
                            "donor": [f"d{i}" for i in range(6)],
                            "group": ["control", "control", "treated", "treated", "mimic",
                                      "exclude"]})
    cells = pd.DataFrame({"cell": [f"c{i}" for i in range(n_cells)],
                          "sample": rng.choice(samples["sample"], n_cells),
                          "population": rng.choice(["CD4 T", "B", "NK", "other"], n_cells)})
    data = tmp_path / "dense.h5ad"
    ad.AnnData(X=x, obs=pd.DataFrame(index=cells["cell"].tolist()),
               var=pd.DataFrame(index=feats)).write_h5ad(data)
    write_view(tmp_path / "v", genes=genes, samples=samples, cells=cells, spec={})
    view, problems = load_view(tmp_path / "v", kind="scrna", vocab=VOCAB)
    assert view is not None, problems
    table = site_table(view, data, kind="scrna")

    assert table.genes == sorted({f"G{j}" for j in range(20)})
    kept = cells[(cells["population"] != "other") & (cells["sample"] != "s5")]
    for row, unit in zip(table.values, table.units.itertuples()):
        sample, pop = unit.unit.split("|")
        mask = cells["cell"].isin(kept.loc[(kept["sample"] == sample)
                                           & (kept["population"] == pop), "cell"]).to_numpy()
        raw = x[mask].sum(axis=0)
        by_symbol = pd.Series(raw, index=[genes[f] for f in feats]).groupby(level=0).sum()
        expected = np.log2(by_symbol[table.genes].to_numpy() / raw.sum() * 1e6 + 1)
        np.testing.assert_allclose(row, expected)
        assert unit.n_cells == mask.sum()


def test_view_ids_missing_from_the_data_give_an_empty_table_not_an_error(tmp_path):
    no_genes = _scrna_table(tmp_path / "a", genes={"GZMB": "GZMB", "CD3E": "CD3E"})
    assert no_genes.genes == [] and no_genes.values.shape == (4, 0)
    np.testing.assert_array_equal(no_genes.population_mean_cpm("B"), np.zeros(0))
    renamed = _cells_frame().assign(cell=lambda d: "x" + d["cell"])
    no_cells = _scrna_table(tmp_path / "b", cells=renamed)
    assert len(no_cells.units) == 0 and no_cells.values.shape == (0, 2)
    assert len(no_cells.composition) == 0


BULK = pd.DataFrame({
    "native_id": ["P1", "P2", "P3", "P4", "P5", "0006"],
    "b1": [1, 0, 5, 7, 1, 3], "b2": [2, 10, 5, 7, np.nan, 3],
    "b3": [3, 20, 1, 7, 1, 3], "b4": [4, 30, 9, 7, 1, 3],
    "b5": [100, 15, 0, 0, 0, 0], "b6": [100, 15, 0, 0, 0, 0],   # excluded samples
    "b7": [0, 0, 0, 0, 0, 0]})                                  # not in samples.tsv
#: P1/P2 share a symbol: over the kept samples b1-b4 P2 has the larger IQR (15 vs 1.5); with the
#: excluded b5/b6 included P1 would win (73.75 vs 7.5). P4 is unmapped, P5 has a missing value,
#: "0006" is a numeric-looking id that must stay a string.
BULK_GENES = {"P1": "IFI44L", "P2": "ifi44l", "P3": "GBP5", "P4": "", "P5": "STAT1",
              "0006": "CONST"}
BULK_SAMPLES = pd.DataFrame({"sample": [f"b{i}" for i in range(1, 7)],
                             "donor": [f"p{i}" for i in range(1, 7)],
                             "group": ["control", "control", "treated", "treated", "exclude",
                                       "exclude"]})


def _bulk_table(tmp_path: Path, matrix: pd.DataFrame, genes: dict[str, str],
                samples: pd.DataFrame):
    tmp_path.mkdir(parents=True, exist_ok=True)
    data = tmp_path / "expr.tsv"
    matrix.to_csv(data, sep="\t", index=False)
    write_view(tmp_path / "view", genes=genes, samples=samples, cells=None,
               spec={"platform": "HT-12"})
    view, problems = load_view(tmp_path / "view", kind="bulk", vocab=VOCAB)
    assert view is not None and view.cells is None, problems
    return site_table(view, data, kind="bulk")


def test_bulk_keeps_the_largest_iqr_probe_and_rank_normalizes(tmp_path):
    table = _bulk_table(tmp_path, BULK, BULK_GENES, BULK_SAMPLES)
    assert table.kind == "bulk" and table.mean_cpm is None and table.composition is None
    assert table.genes == ["CONST", "GBP5", "IFI44L"]
    assert table.units.to_dict("list") == {
        "unit": ["b1", "b2", "b3", "b4"], "donor": ["p1", "p2", "p3", "p4"],
        "group": ["control", "control", "treated", "treated"], "population": ["all"] * 4,
        "n_cells": [0] * 4}
    expected = np.column_stack([
        np.zeros(4),                                            # constant: all ties -> 0
        norm.ppf(np.array([2.5, 2.5, 1, 4]) / 5),               # P3, ties averaged
        norm.ppf(np.array([1, 2, 3, 4]) / 5)])                  # P2, not P1
    np.testing.assert_allclose(table.values, expected)
    with pytest.raises(ValueError):
        table.population_mean_cpm("all")


def test_rank_normal_is_scale_free_monotone_and_centred(tmp_path):
    rng = np.random.default_rng(1)
    raw = rng.lognormal(5, 1, size=(8, 10))                     # probes x samples, no ties
    probes = [f"P{i}" for i in range(8)]
    samples = pd.DataFrame({"sample": [f"b{j}" for j in range(10)],
                            "donor": [f"p{j}" for j in range(10)],
                            "group": ["control"] * 5 + ["treated"] * 5})
    genes = {p: f"G{i}" for i, p in enumerate(probes)}
    linear = pd.DataFrame(raw, columns=samples["sample"]).assign(native_id=probes)
    logged = pd.DataFrame(np.log2(raw), columns=samples["sample"]).assign(native_id=probes)
    a = _bulk_table(tmp_path / "lin", linear[["native_id", *samples["sample"]]], genes, samples)
    b = _bulk_table(tmp_path / "log", logged[["native_id", *samples["sample"]]], genes, samples)
    np.testing.assert_allclose(a.values, b.values)              # platform scale does not matter
    np.testing.assert_allclose(a.values.sum(axis=0), 0.0, atol=1e-12)
    for j in range(8):
        assert (np.argsort(a.values[:, j]) == np.argsort(raw[j])).all()


def test_bulk_samples_missing_from_the_data_give_an_empty_table(tmp_path):
    renamed = BULK_SAMPLES.assign(sample=lambda d: "x" + d["sample"])
    table = _bulk_table(tmp_path, BULK, BULK_GENES, renamed)
    assert len(table.units) == 0 and table.genes == [] and table.values.shape == (0, 0)
