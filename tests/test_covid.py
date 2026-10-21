"""Case 1 (COVID) site build on a toy raw file: offline, no downloads, no agents."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from agentcoop.cases import covid
from agentcoop.coop.claims import parse_claims

BERLIN = next(s for s in covid.SITES if s.name == "Berlin")


def _raw(path) -> None:
    """Berlin's layout in miniature: two samples of one severe donor (early and late), a mild
    donor, a control; harmonized columns that must not reach the site file."""
    samples = [("D1", "S_D1_d16", "severe_late"), ("D1", "S_D1_d9", "severe_early"),
               ("D2", "S_D2_d7", "mild_early"), ("H1", "S_H1_d0", "control")]
    rows = [(d, s, c, lab) for d, s, c in samples for lab in ("Classical Monocytes", "NK")
            for _ in range(5)]
    obs = pd.DataFrame(rows, columns=["donor_id", "sample", "COVID-19 Condition",
                                      "cluster_labels_res.0.4"])
    obs["Cohort"] = ["Reyes" if d == "H1" else "1" for d in obs["donor_id"]]
    obs["assay"] = ["10x 3' v2" if d == "H1" else "10x 3' v3" for d in obs["donor_id"]]
    obs["sex"], obs["development_stage"] = "female", "adult stage"
    obs["Cell.group"] = "harmonized"                      # Tian's harmonized label: dropped
    obs["disease_general"] = "harmonized"
    obs.index = [f"bc{i}" for i in range(len(obs))]
    counts = sp.csr_matrix(np.random.default_rng(0).poisson(2.0, (len(obs), 6)).astype(np.float32))
    var = pd.DataFrame({"feature_name": [f"G{i}" for i in range(6)]},
                       index=[f"ENSG{i:011d}" for i in range(6)])
    a = ad.AnnData(X=counts / 2, obs=obs.astype("category"), var=var)
    a.raw = ad.AnnData(X=counts, var=var)
    a.write_h5ad(path)


@pytest.fixture
def raw(tmp_path):
    _raw(tmp_path / BERLIN.file)
    return tmp_path


def test_one_sample_per_donor_highest_severity_then_earliest(raw):
    src = ad.read_h5ad(raw / BERLIN.file)
    assert covid.chosen_samples(src.obs, BERLIN) == {"D1": "S_D1_d9", "D2": "S_D2_d7",
                                                     "H1": "S_H1_d0"}


def test_site_file_keeps_raw_counts_and_whitelisted_columns_only(raw):
    adata, card = covid.build_site(BERLIN, raw)
    assert adata.n_obs == 30 and adata.obs_names.is_unique
    assert set(adata.obs.columns) == {"sample", "donor", "condition", "assay", "sex", "age",
                                      "cell_label", "cohort"}
    assert list(adata.var.columns) == ["feature_name"] and adata.var_names[0].startswith("ENSG")
    assert np.allclose(adata.X.data, np.round(adata.X.data))           # counts, not X/2
    text = " ".join(map(str, adata.obs.astype(str).to_numpy().ravel())) + str(card)
    for leaked in ("D1", "S_D1", "bc0", "harmonized"):
        assert leaked not in text
    assert set(card["codes"]) == {"severe_early", "mild_early", "control"}
    assert card["donors_per_code"] == {"control": 1, "mild_early": 1, "severe_early": 1}
    assert all(c in covid.TOOL_GROUPS["Berlin"] for c in card["codes"])


def test_composition_only_populations_reject_de_claims():
    item = {"kind": "de", "population": "pDC", "contrast": ["severe", "healthy"],
            "up_genes": ["IFI27", "ISG15", "MX1"], "down_genes": [], "effect": 1.0, "rank": 1}
    claims, errors = parse_claims([item, {**item, "kind": "composition", "up_genes": []}],
                                  site="Berlin", vocab=covid.CASE_VOCAB, max_claims=10)
    assert [c.kind for c in claims] == ["composition"]
    assert any("composition claims only" in e for e in errors)
