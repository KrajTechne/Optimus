"""
Validation core: input checking, job expansion and result aggregation. No Modal imports, so it runs
locally (and in tests) as well as inside the Modal driver.

A *design* is one row of the input CSV; a *job* is one design x model x seed; the *tall* table has one
row per finished job; the *summary* has one row per design x model.
"""
import itertools
import re

import numpy as np
import pandas as pd

from StrucTools import binder_binds_all_targets

MODELS = ("of3", "native_af3", "esmfold2")
REQUIRED_COLUMNS = ("design_name", "seq_binder", "seq_target")
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
MAX_IDENTICAL_COPIES = 4          # identical target chains are matched by trying every permutation
SCORE_COLUMNS = ("iptm", "ipsae_min", "binder_plddt", "complex_plddt")
CONSISTENCY_COLUMNS = ("paratope_recall", "paratope_jaccard", "epitope_recall", "epitope_jaccard")


# ---------------------------------------------------------------------------------------------- input

def load_designs(source) -> pd.DataFrame:
    """Read and check the designs CSV (path or DataFrame). Every problem is reported at once, by CSV line."""
    df = pd.read_csv(source, dtype=str, keep_default_na=False) if isinstance(source, str) else source.astype(str)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"designs CSV is missing required column(s): {missing}")

    # A refiner top_designs.csv carries its own interface; treat it as the reference.
    if not any(c.startswith("ref_") for c in df.columns):
        df = df.rename(columns={c: f"ref_{c}" for c in df.columns if re.match(r"(paratope|epitope)_indices_[A-Z]$", c)})

    errors = []
    for line, row in enumerate(df.itertuples(), start=2):      # CSV line numbers: header is line 1
        name = row.design_name.strip()
        if not re.fullmatch(r"[A-Za-z0-9._-]+", name) or name in (".", ".."):
            errors.append(f"line {line}: design_name {name!r} must be letters, digits, '.', '_' or '-' (it becomes a folder name)")
        chains = {"seq_binder": [row.seq_binder], "seq_target": row.seq_target.split(",")}
        for column, seqs in chains.items():
            for seq in seqs:
                if not seq or set(seq) - AMINO_ACIDS:
                    bad = sorted(set(seq) - AMINO_ACIDS) or "empty sequence"
                    errors.append(f"line {line} ({name}): {column} has invalid content {bad}")
    duplicated = df.design_name[df.design_name.duplicated()].unique().tolist()
    if duplicated:
        errors.append(f"design_name must be unique, repeated: {duplicated}")
    if errors:
        raise ValueError("invalid designs CSV:\n  " + "\n  ".join(errors))
    return df.reset_index(drop=True)


def expand_jobs(designs: pd.DataFrame, models, seeds, num_samples: int = 5, max_jobs: int = 200) -> list[dict]:
    """design x model x seed jobs, refusing before anything runs if there are too many or the names are wrong."""
    unknown = [m for m in models if m not in MODELS]
    if unknown or not models:
        raise ValueError(f"models must be a non-empty subset of {MODELS}, got {list(models)}")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds must be a non-empty list without repeats, got {list(seeds)}")
    if num_samples < 1:
        raise ValueError(f"num_samples must be at least 1, got {num_samples}")
    n_jobs = len(designs) * len(models) * len(seeds)
    if n_jobs > max_jobs:
        raise ValueError(f"{len(designs)} designs x {len(models)} models x {len(seeds)} seeds = {n_jobs} jobs, "
                         f"above --max_jobs {max_jobs}")
    return [{"design_name": row.design_name, "seq_binder": row.seq_binder, "seq_target": row.seq_target,
             "ligands": getattr(row, "ligands", ""), "model": model, "seed": int(seed),
             "num_samples": int(num_samples)}
            for row in designs.itertuples() for model in models for seed in seeds]


def job_folder(design_name: str, model: str, seed: int) -> str:
    return f"{design_name}/{model}_seed{seed}"


# ------------------------------------------------------------------------------------ interface overlap

def _interface(indices: dict, prefix: str, letters: str) -> set:
    """{(chain, position)} over all target chains from `<prefix>_<chain>` comma-separated index strings."""
    return {(chain, int(pos)) for chain in letters for pos in str(indices.get(f"{prefix}_{chain}", "")).split(",") if pos.strip()}


def _permutations(seq_target: str):
    """Chain relabelings (pred chain -> ref chain) that only swap identical target copies; identity first."""
    seqs = seq_target.split(",")
    letters = [chr(ord("B") + i) for i in range(len(seqs))]
    groups = [[l for l, s in zip(letters, seqs) if s == seq] for seq in dict.fromkeys(seqs)]
    # More copies than the cap: no swapping for that group (permutations grow factorially).
    groups = [x for g in groups for x in ([g] if len(g) <= MAX_IDENTICAL_COPIES else [[l] for l in g])]
    options = [[dict(zip(g, p)) for p in itertools.permutations(g)] for g in groups]
    for combo in itertools.product(*options):
        yield {k: v for part in combo for k, v in part.items()}


def _overlap(ref: set, pred: set) -> tuple[float, float]:
    """(recall of ref, Jaccard); NaN when the reference is empty."""
    if not ref:
        return np.nan, np.nan
    return len(ref & pred) / len(ref), len(ref & pred) / len(ref | pred)


def interface_consistency(ref: dict, pred: dict, seq_target: str) -> dict:
    """
    How well `pred`'s paratope/epitope reproduces `ref`'s, both as `{paratope|epitope}_indices_<chain>` strings
    in sequence-position numbering. Identical target copies may come back with swapped labels, so the
    relabeling with the best combined (paratope + epitope) Jaccard is used, one relabeling for both sides.
    """
    letters = "".join(chr(ord("B") + i) for i in range(len(seq_target.split(","))))
    ref_para, ref_epi = _interface(ref, "paratope_indices", letters), _interface(ref, "epitope_indices", letters)
    if not ref_para and not ref_epi:
        return {}
    pred_para, pred_epi = _interface(pred, "paratope_indices", letters), _interface(pred, "epitope_indices", letters)
    best = None
    for relabel in _permutations(seq_target):
        para = {(relabel[c], p) for c, p in pred_para}
        epi = {(relabel[c], p) for c, p in pred_epi}
        scores = (*_overlap(ref_para, para), *_overlap(ref_epi, epi))
        total = np.nansum([scores[1], scores[3]])
        if best is None or total > best[0] + 1e-12:
            best = (total, scores, relabel)
    _, (para_recall, para_jaccard, epi_recall, epi_jaccard), relabel = best
    moved = ";".join(f"{k}->{v}" for k, v in relabel.items() if k != v)
    return {"paratope_recall": para_recall, "paratope_jaccard": para_jaccard,
            "epitope_recall": epi_recall, "epitope_jaccard": epi_jaccard, "chain_assignment": moved or "identity"}


def add_interface_consistency(tall: pd.DataFrame, designs: pd.DataFrame) -> pd.DataFrame:
    """Append consistency columns (reference = each design's `ref_*` columns); empty for designs without one."""
    by_design = designs.set_index("design_name")
    rows = []
    for row in tall.to_dict(orient="records"):
        design = by_design.loc[row["design_name"]]
        ref = {c[len("ref_"):]: design[c] for c in by_design.columns if c.startswith("ref_")}
        rows.append(interface_consistency(ref, row, design.seq_target))
    return pd.concat([tall.reset_index(drop=True), pd.DataFrame(rows, columns=[*CONSISTENCY_COLUMNS, "chain_assignment"])], axis=1)


def add_contact_check(tall: pd.DataFrame, designs: pd.DataFrame, epitope_residues: str = "", paratope_residues: str = "") -> pd.DataFrame:
    """
    `contact_check_passed` per job, from the predicted indices with the same rule the refiner uses. Done here
    rather than inside the jobs so the desired residues can be changed and re-scored without any GPU time.
    """
    num_targets = designs.set_index("design_name").seq_target.str.count(",") + 1
    passed = [binder_binds_all_targets(row, num_targets[row["design_name"]], epitope_residues, paratope_residues)
              for row in tall.to_dict(orient="records")]
    return tall.assign(contact_check_passed=passed)


# ------------------------------------------------------------------------------------------ aggregation

def summarize(tall: pd.DataFrame, designs: pd.DataFrame) -> pd.DataFrame:
    """
    One row per design x model. Scores are per-seed best-of-samples values (best by iptm), so means/maxima
    are biased upward by the sample selection. Seed-to-seed overlap (2+ seeds) is the model's own noise floor
    for the interface-consistency columns.
    """
    seq_target = designs.set_index("design_name").seq_target
    rows = []
    for (design, model), group in tall.groupby(["design_name", "model"], sort=False):
        out = {"design_name": design, "model": model, "n_seeds": len(group)}
        for column in SCORE_COLUMNS:
            if column in group:
                out |= {f"{column}_mean": group[column].mean(), f"{column}_min": group[column].min(),
                        f"{column}_max": group[column].max()}
        if "contact_check_passed" in group:
            out["contact_pass_fraction"] = group["contact_check_passed"].astype(bool).mean()
        for column in CONSISTENCY_COLUMNS:
            if column in group:
                out[f"{column}_mean"] = group[column].mean()
        pairs = [interface_consistency(a, b, seq_target[design])
                 for a, b in itertools.combinations(group.to_dict(orient="records"), 2)]
        pairs = [p for p in pairs if p]
        if pairs:
            out["seed_paratope_jaccard"] = np.nanmean([p["paratope_jaccard"] for p in pairs])
            out["seed_epitope_jaccard"] = np.nanmean([p["epitope_jaccard"] for p in pairs])
        rows.append(out)
    return pd.DataFrame(rows)
