"""Site views: the machine-readable handoff a site agent writes, and the table testers read.

A site agent describes its private data in fixed files (spec §2.2): ``genes.tsv`` (native
feature id -> symbol), ``samples.tsv`` (sample -> donor, case group, covariates), ``cells.tsv``
(scRNA only: cell -> sample, population) and ``spec.json`` (its analysis spec). ``load_view``
checks them against the case vocabulary and returns every defect as text instead of raising: a
malformed view is a finding about the site (a handoff loss step 4 can repair), not a crash.

``site_table`` turns a valid view plus the site's raw data into the only object testers read,
one row per unit: (sample, population) pseudobulk for scRNA, sample for bulk. Ids in the view
that the data lacks are dropped, so a mismatched view yields few genes or units and the testers
answer ``untestable`` — the signal that triggers resolution — rather than an exception.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import norm, rankdata

from agentcoop.coop.claims import Vocabulary

KINDS = ("scrna", "bulk")
#: Sample group for samples in the data but outside every contrast.
EXCLUDE = "exclude"
#: Cell population outside the case vocabulary (kept in the view, never in a unit).
OTHER = "other"
REQUIRED = {"genes.tsv": ("native_id", "symbol"),
            "samples.tsv": ("sample", "donor", "group"),
            "cells.tsv": ("cell", "sample", "population")}
#: Offending values quoted per message; messages go into repair prompts, so they stay short.
SHOW = 5


@dataclass
class SiteView:
    root: Path
    genes: dict[str, str]           # native feature id -> symbol (uppercase, unmapped rows dropped)
    samples: pd.DataFrame           # columns: sample, donor, group (+ covariates), all strings
    cells: pd.DataFrame | None      # columns: cell, sample, population (scRNA only)
    spec: dict


def _few(values) -> str:
    vals = sorted({str(v) for v in values})
    more = f" (+{len(vals) - SHOW} more)" if len(vals) > SHOW else ""
    return ", ".join(repr(v) for v in vals[:SHOW]) + more


def _read_tsv(path: Path, errors: list[str]) -> pd.DataFrame | None:
    """Strict TSV: every row has the header's field count, every cell a stripped string.

    Parsed with ``csv`` rather than pandas, which silently truncates ragged rows or turns the
    first column into an index; an absent, unreadable or ragged file becomes an error message.
    """
    if not path.is_file():
        errors.append(f"{path.name}: missing")
        return None
    try:
        rows = [r for r in csv.reader(io.StringIO(path.read_text(encoding="utf-8")),
                                         delimiter="\t") if r]
    except (OSError, ValueError, csv.Error) as e:      # ValueError covers UnicodeDecodeError
        errors.append(f"{path.name}: unreadable ({type(e).__name__}: {e})")
        return None
    if not rows:
        errors.append(f"{path.name}: empty")
        return None
    header = [c.strip() for c in rows[0]]
    problems = []
    dup = {c for c in header if header.count(c) > 1}
    if dup:
        problems.append(f"duplicate column(s) {_few(dup)}")
    ragged = [i for i, r in enumerate(rows[1:], start=2) if len(r) != len(header)]
    if ragged:
        problems.append(f"{len(ragged)} row(s) whose field count differs from the header "
                        f"(first: line {ragged[0]})")
    missing = [c for c in REQUIRED[path.name] if c not in header]
    if missing:
        problems.append(f"missing column(s) {', '.join(missing)} (has: {', '.join(header)})")
    if problems:
        errors.extend(f"{path.name}: {p}" for p in problems)
        return None
    return pd.DataFrame(rows[1:], columns=header).apply(lambda s: s.str.strip())


def _check_ids(df: pd.DataFrame, col: str, name: str, errors: list[str], *, unique: bool) -> None:
    """Ids must be non-empty and, where they key a table, unique."""
    empty = int((df[col] == "").sum())
    if empty:
        errors.append(f"{name}: {empty} row(s) with an empty {col}")
    if unique:
        dup = df.loc[df[col].duplicated() & (df[col] != ""), col]
        if len(dup):
            errors.append(f"{name}: duplicate {col} {_few(dup)}")


def _read_spec(path: Path, errors: list[str]) -> dict | None:
    if not path.is_file():
        errors.append("spec.json: missing")
        return None
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:      # JSONDecodeError, UnicodeDecodeError
        errors.append(f"spec.json: unreadable ({type(e).__name__}: {e})")
        return None
    if not isinstance(spec, dict):
        errors.append(f"spec.json: must be a JSON object, got {type(spec).__name__}")
        return None
    return spec


def load_view(root: Path, *, kind: str, vocab: Vocabulary) -> tuple[SiteView | None, list[str]]:
    """``(view, warnings)`` when the view is usable, else ``(None, errors)``.

    Never raises on the site's files; every defect is collected (not just the first) so a single
    repair call can fix them all. Only an invalid ``kind`` (a caller bug) raises.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown data kind {kind!r}")
    root = Path(root)
    if not root.is_dir():
        return None, [f"view directory {root} missing"]
    errors: list[str] = []
    warnings: list[str] = []

    genes = _read_tsv(root / "genes.tsv", errors)
    mapping: dict[str, str] = {}
    if genes is not None:
        _check_ids(genes, "native_id", "genes.tsv", errors, unique=True)
        symbols = genes["symbol"].str.upper()
        mapped = symbols != ""
        if (~mapped).any():
            warnings.append(f"genes.tsv: {int((~mapped).sum())} row(s) without a symbol dropped")
        mapping = dict(zip(genes.loc[mapped, "native_id"], symbols[mapped]))
        if not mapping:
            errors.append("genes.tsv: no feature mapped to a symbol")

    samples = _read_tsv(root / "samples.tsv", errors)
    if samples is not None:
        _check_ids(samples, "sample", "samples.tsv", errors, unique=True)
        _check_ids(samples, "donor", "samples.tsv", errors, unique=False)
        allowed = (*vocab.groups, EXCLUDE)
        bad = samples.loc[~samples["group"].isin(allowed), "group"]
        if len(bad):
            errors.append(f"samples.tsv: group(s) {_few(bad)} not in the vocabulary "
                          f"(allowed: {', '.join(allowed)})")
        elif not (samples["group"] != EXCLUDE).any():
            errors.append("samples.tsv: no sample in a vocabulary group")

    cells = None
    if kind == "scrna":
        cells = _read_tsv(root / "cells.tsv", errors)
        if cells is not None:
            _check_ids(cells, "cell", "cells.tsv", errors, unique=True)
            allowed = (*vocab.populations, OTHER)
            bad = cells.loc[~cells["population"].isin(allowed), "population"]
            if len(bad):
                errors.append(f"cells.tsv: population(s) {_few(bad)} not in the vocabulary "
                              f"(allowed: {', '.join(allowed)})")
            elif not (cells["population"] != OTHER).any():
                errors.append("cells.tsv: no cell in a vocabulary population")
            if samples is not None:
                unknown = cells.loc[~cells["sample"].isin(samples["sample"]), "sample"]
                if len(unknown):
                    errors.append(f"cells.tsv: {len(unknown)} cell(s) reference unknown "
                                  f"sample(s) {_few(unknown)}")

    spec = _read_spec(root / "spec.json", errors)
    if errors:
        return None, errors
    return SiteView(root=root, genes=mapping, samples=samples.reset_index(drop=True),
                    cells=None if cells is None else cells.reset_index(drop=True),
                    spec=spec), warnings


def write_view(root: Path, *, genes: dict[str, str], samples: pd.DataFrame,
               cells: pd.DataFrame | None, spec: dict) -> None:
    """Write the view files in the format site agents are asked for (case scripts, tests)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"native_id": list(genes), "symbol": list(genes.values())}).to_csv(
        root / "genes.tsv", sep="\t", index=False)
    samples.to_csv(root / "samples.tsv", sep="\t", index=False)
    if cells is not None:
        cells.to_csv(root / "cells.tsv", sep="\t", index=False)
    (root / "spec.json").write_text(json.dumps(spec, indent=1, sort_keys=True), encoding="utf-8")


@dataclass
class SiteTable:
    """What testers read. Rows of ``values`` follow ``units``; columns follow ``genes`` (sorted).

    units: unit, donor, group, population ("all" for bulk), n_cells (0 for bulk).
    values: scRNA log2(CPM+1) pseudobulk per (sample, population); bulk per-gene rank-to-normal.
    mean_cpm: scRNA only, populations (sorted) x genes, mean CPM over each population's units.
    composition: scRNA only, one row per sample: fraction of its kept cells in each population,
    plus "group" and "donor".
    """
    kind: str
    genes: list[str]
    units: pd.DataFrame
    values: np.ndarray
    mean_cpm: np.ndarray | None
    composition: pd.DataFrame | None

    def population_mean_cpm(self, population: str) -> np.ndarray:
        """Mean CPM per gene over the population's units.

        A population the site lacks gets zeros: no gene is measurable there, so a test of it is
        untestable, never supported.
        """
        if self.mean_cpm is None:
            raise ValueError("bulk site tables carry no CPM")
        pops = sorted(self.units["population"].unique())
        if population not in pops:
            return np.zeros(len(self.genes))
        return self.mean_cpm[pops.index(population)]


def site_table(view: SiteView, data_path: Path, *, kind: str) -> SiteTable:
    """Compute the site table of a valid view (once per view; testers read nothing else)."""
    if kind == "scrna":
        return _scrna_table(view, Path(data_path))
    if kind == "bulk":
        return _bulk_table(view, Path(data_path))
    raise ValueError(f"unknown data kind {kind!r}")


def _kept_samples(view: SiteView) -> pd.DataFrame:
    return view.samples[view.samples["group"] != EXCLUDE].set_index("sample")


def _scrna_table(view: SiteView, data_path: Path) -> SiteTable:
    """Pseudobulk by sparse indicator products: cells -> units, native features -> symbols.

    The library size of a unit is its total count over *all* native features, so how many
    features the agent mapped does not change the CPM of the genes it did map.
    """
    import anndata as ad    # heavy; only scRNA sites need it

    adata = ad.read_h5ad(data_path)
    samples = _kept_samples(view)
    cells = view.cells[(view.cells["population"] != OTHER)
                       & view.cells["sample"].isin(samples.index)]
    rows = pd.Index(adata.obs_names).get_indexer(cells["cell"])
    cells, rows = cells[rows >= 0], rows[rows >= 0]

    grouped = cells.groupby(["sample", "population"], sort=True)
    codes = grouped.ngroup().to_numpy()
    units = grouped.size().rename("n_cells").reset_index()
    to_unit = sp.csr_matrix((np.ones(len(rows)), (codes, rows)), shape=(len(units), adata.n_obs))
    counts = (to_unit @ sp.csr_matrix(adata.X)).tocsr()          # units x native features
    library = np.asarray(counts.sum(axis=1), dtype=float).ravel()

    cols = pd.Index(adata.var_names).get_indexer(list(view.genes))
    found = cols >= 0
    symbols, sym_codes = np.unique(np.array(list(view.genes.values()), dtype=str)[found],
                                   return_inverse=True)
    to_symbol = sp.csr_matrix((np.ones(int(found.sum())), (cols[found], sym_codes)),
                              shape=(adata.n_vars, len(symbols)))
    cpm = (counts @ to_symbol).toarray() / np.maximum(library, 1.0)[:, None] * 1e6

    pops = sorted(units["population"].unique())
    mean_cpm = np.array([cpm[(units["population"] == p).to_numpy()].mean(axis=0) for p in pops])
    tab = pd.crosstab(cells["sample"], cells["population"])
    composition = tab.div(tab.sum(axis=1), axis=0)
    composition.columns.name = None
    composition["group"] = composition.index.map(samples["group"])
    composition["donor"] = composition.index.map(samples["donor"])

    units.insert(0, "unit", units["sample"] + "|" + units["population"])
    units["donor"] = units["sample"].map(samples["donor"])
    units["group"] = units["sample"].map(samples["group"])
    return SiteTable(kind="scrna", genes=symbols.tolist(),
                     units=units[["unit", "donor", "group", "population", "n_cells"]],
                     values=np.log2(cpm + 1.0),
                     mean_cpm=mean_cpm.reshape(len(pops), len(symbols)),
                     composition=composition)


def _bulk_table(view: SiteView, data_path: Path) -> SiteTable:
    """One probe per symbol (largest IQR across the kept samples — frozen rule), then a per-gene
    van der Waerden rank-to-normal transform, so sites on different platforms and scales
    (raw linear, log2, RNA-seq) give comparable scores."""
    header = pd.read_csv(data_path, sep="\t", nrows=0).columns
    samples = _kept_samples(view)
    kept = [s for s in samples.index if s in header[1:]]
    data = pd.read_csv(data_path, sep="\t", usecols=[header[0], *kept], index_col=header[0],
                       dtype={header[0]: str})          # ids stay strings ("0012" is not 12)

    x = data.loc[data.index.isin(list(view.genes)), kept].apply(pd.to_numeric, errors="coerce")
    arr = x.to_numpy(dtype=float)
    complete = np.isfinite(arr).all(axis=1) & bool(kept)       # probes with a value everywhere
    x, arr = x[complete], arr[complete]
    iqr = np.subtract(*np.percentile(arr, [75, 25], axis=1)) if arr.size else np.zeros(0)
    probes = pd.DataFrame({"native_id": x.index, "symbol": x.index.map(view.genes), "iqr": iqr})
    probes = (probes.sort_values(["symbol", "iqr", "native_id"], ascending=[True, False, True])
              .drop_duplicates("symbol"))
    expr = x.loc[probes["native_id"]].to_numpy(dtype=float).T     # samples x symbols
    values = norm.ppf(rankdata(expr, axis=0) / (len(kept) + 1))

    units = pd.DataFrame({"unit": kept, "donor": samples.loc[kept, "donor"].tolist(),
                          "group": samples.loc[kept, "group"].tolist(),
                          "population": ["all"] * len(kept), "n_cells": [0] * len(kept)})
    return SiteTable(kind="bulk", genes=probes["symbol"].tolist(), units=units,
                     values=values.reshape(len(kept), len(probes)), mean_cpm=None,
                     composition=None)
