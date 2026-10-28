"""
Validation core: input checking, job expansion and result aggregation. No Modal imports, so it runs
locally (and in tests) as well as inside the Modal driver.

A *design* is one row of the input CSV; a *job* is one design x model x seed; the *tall* table has one
row per finished job; the *summary* has one row per design x model.
"""
import itertools
import os
import re

import numpy as np
import pandas as pd

from StrucTools import binder_binds_all_targets, binder_rmsds, protein_ca_atoms

MODELS = ("of3", "native_af3", "esmfold2")
REQUIRED_COLUMNS = ("design_name", "seq_binder", "seq_target")
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
MAX_IDENTICAL_COPIES = 4          # identical target chains are matched by trying every permutation
SCORE_COLUMNS = ("iptm", "ipsae_min", "binder_plddt", "binder_plddt70_fraction", "complex_plddt", "rmsd_fold")
CONFIDENT_PLDDT = 70   # "confident" residue, the usual AlphaFold cutoff
POSE_PREFIX = "rmsd_pose_target_"   # one column per target chain: rmsd_pose_target_B, ...
CONSISTENCY_COLUMNS = ("paratope_recall", "paratope_jaccard", "epitope_recall", "epitope_jaccard")


# ---------------------------------------------------------------------------------------------- input

def load_designs(source) -> pd.DataFrame:
    """Read and check the designs CSV (path or DataFrame). Every problem is reported at once, by CSV line."""
    df = pd.read_csv(source, dtype=str, keep_default_na=False) if isinstance(source, str) else source.astype(str)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"designs CSV is missing required column(s): {missing}")

    # A refiner top_designs.csv carries its own interface and structure; treat them as the reference.
    if not any(re.match(r"ref_(paratope|epitope)_indices_[A-Z]$", c) for c in df.columns):
        df = df.rename(columns={c: f"ref_{c}" for c in df.columns if re.match(r"(paratope|epitope)_indices_[A-Z]$", c)})
    if "path_structure_ref" not in df.columns and "path_structure" in df.columns:
        df = df.assign(path_structure_ref=df.path_structure)

    errors = []
    for line, row in enumerate(df.itertuples(), start=2):      # CSV line numbers: header is line 1
        name = row.design_name.strip()
        if not re.fullmatch(r"[A-Za-z0-9._-]+", name) or name in (".", ".."):
            errors.append(f"line {line}: design_name {name!r} must be letters, digits, '.', '_' or '-' (it becomes a folder name)")
        ref_chains = [c.strip() for c in getattr(row, "ref_chain_ids", "").split(",") if c.strip()]
        expected = 1 + len(row.seq_target.split(","))
        if ref_chains and (len(ref_chains) != expected or len(set(ref_chains)) != expected):
            errors.append(f"line {line} ({name}): ref_chain_ids {ref_chains} must be {expected} different chain IDs "
                          f"(the binder, then each of the {expected - 1} target chain(s))")
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


def _best_rmsds(ref_ca, ref_chains: list, pred_ca, seq_binder: str, seq_target: str) -> dict:
    """binder_rmsds() under the target-copy relabeling (identical copies only) with the lowest total pose RMSD."""
    sequences = [seq_binder, *seq_target.split(",")]
    letters = [chr(ord("B") + i) for i in range(len(ref_chains) - 1)]
    best = None
    for relabel in _permutations(seq_target):                  # pred chain -> ref chain label
        by_ref = {ref: pred for pred, ref in relabel.items()}
        result = binder_rmsds(ref_ca, pred_ca, ref_chains, ["A", *(by_ref[letter] for letter in letters)], sequences)
        total = sum(v for k, v in result.items() if k.startswith(POSE_PREFIX))
        if best is None or total < best[0] - 1e-9:
            best = (total, result)
    return best[1]


def add_structure_rmsd(tall: pd.DataFrame, designs: pd.DataFrame) -> pd.DataFrame:
    """
    `rmsd_fold` and `rmsd_pose_target_<X>` per job, against each design's `path_structure_ref` (reference chains
    from `ref_chain_ids`, blank = A, B, C, ... as in a refiner structure) using the job's predicted `path_structure`.
    Every chain used must have exactly the design's sequence in both structures. Blank when a design has no
    reference; a file that is missing or does not match the design gives blanks and a warning rather than
    failing the run. Computed driver-side so it needs no GPU and can be re-run.
    """
    by_design = designs.set_index("design_name")
    reference, warned, rows = {}, set(), []
    for row in tall.to_dict(orient="records"):
        design = by_design.loc[row["design_name"]]
        path_ref = str(design.get("path_structure_ref", "")).strip()
        path_pred = str(row.get("path_structure", "")).strip()
        if not path_ref or not path_pred:
            rows.append({})
            continue
        try:
            num_targets = len(design.seq_target.split(","))
            ref_chains = ([c.strip() for c in str(design.get("ref_chain_ids", "")).split(",") if c.strip()]
                          or [chr(ord("A") + i) for i in range(num_targets + 1)])
            if row["design_name"] not in reference:
                reference[row["design_name"]] = protein_ca_atoms(path_ref)
            rows.append(_best_rmsds(reference[row["design_name"]], ref_chains, protein_ca_atoms(path_pred),
                                     design.seq_binder, design.seq_target))
        except Exception as exc:
            if (row["design_name"], str(exc)) not in warned:
                warned.add((row["design_name"], str(exc)))
                print(f"WARNING: no RMSD for {row['design_name']} ({row.get('model', '')} seed {row.get('seed', '')}): {exc}")
            rows.append({})
    return pd.concat([tall.reset_index(drop=True), pd.DataFrame(rows, index=range(len(rows)))], axis=1)


def add_binder_plddt70_fraction(tall: pd.DataFrame) -> pd.DataFrame:
    """
    `binder_plddt70_fraction` per job: the share of binder (chain A) residues whose pLDDT is at least 70, read from
    the predicted structure's B-factors (every validation model stores pLDDT there). A prediction in which little of
    the binder is confident has no meaningful RMSD. Blank when the structure cannot be read.
    """
    fractions = []
    for path in tall.get("path_structure", pd.Series("", index=tall.index)):
        try:
            ca = protein_ca_atoms(str(path), load_b_factor=True)
            plddt = ca.b_factor[ca.chain_id == "A"]
            plddt = plddt * 100 if plddt.max() <= 1.0 else plddt      # some writers use a 0-1 scale
            fractions.append(float((plddt >= CONFIDENT_PLDDT).mean()))
        except Exception:
            fractions.append(np.nan)
    return tall.assign(binder_plddt70_fraction=fractions)


def plan_reference_uploads(designs: pd.DataFrame, output_dir: str, outputs_mount: str) -> tuple[pd.DataFrame, list]:
    """
    Decide where each design's reference structure will be read from on Modal. A file on this machine is to be
    uploaded to <output_dir>/reference/<design_name><ext> on the outputs volume (`path_structure_ref` is rewritten to
    where it will appear in the container); a path already under the volume mount (a refiner structure) is kept.
    Anything else is an error, listed by CSV line. Returns (designs with rewritten paths, [(local, volume_relative)]).
    """
    paths, uploads, errors = designs.get("path_structure_ref", pd.Series("", index=designs.index)).tolist(), [], []
    mount = outputs_mount.rstrip("/") + "/"
    for line, (name, path) in enumerate(zip(designs.design_name, paths), start=2):
        path = path.strip()
        if not path:
            continue
        if path.replace("\\", "/").startswith(mount):
            paths[line - 2] = path.replace("\\", "/")
        elif not os.path.isfile(path):
            errors.append(f"line {line} ({name}): path_structure_ref {path!r} is neither a file on this machine nor under {mount} on the volume")
        elif not path.lower().endswith((".pdb", ".cif")):
            errors.append(f"line {line} ({name}): path_structure_ref {path!r} must be a .pdb or .cif file")
        else:
            remote = f"{output_dir}/reference/{name}{os.path.splitext(path)[1].lower()}"
            uploads.append((path, remote))
            paths[line - 2] = f"{mount}{remote}"
    if errors:
        raise ValueError("invalid reference structures:\n  " + "\n  ".join(errors))
    return (designs.assign(path_structure_ref=paths) if "path_structure_ref" in designs else designs), uploads


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
        pose_columns = sorted(c for c in group if c.startswith(POSE_PREFIX))
        for column in [*SCORE_COLUMNS, *pose_columns]:
            if column in group:
                out |= {f"{column}_mean": group[column].mean(), f"{column}_min": group[column].min(),
                        f"{column}_max": group[column].max()}
        if pose_columns:   # the target the binder is placed worst against, by seed-averaged pose RMSD
            out["rmsd_pose_worst_target_mean"] = pd.Series([out[f"{c}_mean"] for c in pose_columns]).max()
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
