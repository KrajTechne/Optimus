import biotite.structure as struc
import biotite.structure.io.pdb as pdb
from biotite.sequence import ProteinSequence
import numpy as np
import pandas as pd
import pytest

import modal_common
from designs_from_structures import designs_from_structures
from StrucTools import binder_rmsds, protein_ca_atoms
from validation_core import add_binder_plddt70_fraction, add_structure_rmsd, load_designs, plan_reference_uploads, summarize

rng = np.random.default_rng(0)
BINDER, TARGET_B, TARGET_C = (rng.normal(scale=8, size=(8, 3)) for _ in range(3))


BINDER_SEQ, TARGET_SEQ = "ACDEFGHI", "GGGGGGGG"


def write_chains(path, chains: dict, seqs: dict = None):
    """chains: {chain_id: (n, 3) CA coordinates}; residues follow `seqs` ({chain_id: sequence}), default the
    binder sequence for chain A or H and glycines for every other chain."""
    seqs = seqs or {}
    atoms = [struc.Atom(xyz, chain_id=chain, res_id=i + 1, atom_name="CA", element="C",
                        res_name=ProteinSequence.convert_letter_1to3((seqs.get(chain) or (BINDER_SEQ if chain in "AH" else TARGET_SEQ))[i]))
             for chain, coords in chains.items() for i, xyz in enumerate(coords)]
    file = pdb.PDBFile()
    file.set_structure(struc.array(atoms))
    file.write(str(path))
    return str(path)


def rotated(coords, degrees=40, shift=(5.0, -2.0, 7.0)):
    a = np.radians(degrees)
    rotation = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return coords @ rotation.T + shift


def rmsds(tmp_path, ref, pred, ref_chains, pred_chains, sequences=None, ref_seqs=None, pred_seqs=None):
    path_ref, path_pred = write_chains(tmp_path / "ref.pdb", ref, ref_seqs), write_chains(tmp_path / "pred.pdb", pred, pred_seqs)
    return binder_rmsds(protein_ca_atoms(path_ref), protein_ca_atoms(path_pred), ref_chains, pred_chains, sequences)


def test_identical_complex_moved_rigidly_has_zero_rmsd(tmp_path):
    ref = {"A": BINDER, "B": TARGET_B}
    pred = {chain: rotated(coords) for chain, coords in ref.items()}
    out = rmsds(tmp_path, ref, pred, ["A", "B"], ["A", "B"])
    assert out == pytest.approx({"rmsd_fold": 0, "rmsd_pose_target_B": 0}, abs=1e-3)


def test_binder_displaced_against_the_target_changes_pose_not_fold(tmp_path):
    ref = {"A": BINDER, "B": TARGET_B}
    pred = {"A": BINDER + [3, 0, 0], "B": TARGET_B}
    out = rmsds(tmp_path, ref, pred, ["A", "B"], ["A", "B"])
    assert out["rmsd_fold"] == pytest.approx(0, abs=1e-3) and out["rmsd_pose_target_B"] == pytest.approx(3, abs=1e-3)


def test_each_target_gets_its_own_pose_rmsd(tmp_path):
    ref = {"A": BINDER, "B": TARGET_B, "C": TARGET_C}
    pred = {"A": BINDER, "B": TARGET_B, "C": TARGET_C + [5, 0, 0]}      # only C moved relative to the binder
    out = rmsds(tmp_path, ref, pred, list("ABC"), list("ABC"))
    assert out["rmsd_pose_target_B"] == pytest.approx(0, abs=1e-3) and out["rmsd_pose_target_C"] == pytest.approx(5, abs=1e-3)


def test_reference_chain_ids_can_differ_from_the_pipeline_ones(tmp_path):
    ref = {"H": BINDER, "L": TARGET_B, "X": TARGET_C}
    pred = {"A": BINDER, "B": TARGET_B, "C": TARGET_C}
    out = rmsds(tmp_path, ref, pred, ["H", "L", "X"], ["A", "B", "C"])
    assert set(out) == {"rmsd_fold", "rmsd_pose_target_B", "rmsd_pose_target_C"} and max(out.values()) < 1e-3


def test_residue_count_mismatch_and_missing_chain_are_errors(tmp_path):
    with pytest.raises(ValueError, match="has 8 residues but predicted chain B has 5"):
        rmsds(tmp_path, {"A": BINDER, "B": TARGET_B}, {"A": BINDER, "B": TARGET_B[:5]}, ["A", "B"], ["A", "B"])
    with pytest.raises(ValueError, match="chain Z has no amino-acid residues in the reference"):
        rmsds(tmp_path, {"A": BINDER, "B": TARGET_B}, {"A": BINDER, "B": TARGET_B}, ["A", "Z"], ["A", "B"])


def test_calcium_ion_is_not_a_residue(tmp_path):
    path = write_chains(tmp_path / "x.pdb", {"A": BINDER})
    atoms = pdb.PDBFile.read(path).get_structure(model=1)
    ion = struc.array([struc.Atom([1, 1, 1], chain_id="A", res_id=99, res_name="CA", atom_name="CA", element="CA", hetero=True)])
    file = pdb.PDBFile()
    file.set_structure(atoms + ion)
    file.write(path)
    assert len(protein_ca_atoms(path)) == 8


# ------------------------------------------------------------------------------------ in the validation table

def designs_and_tall(tmp_path, pred_chains, seq_target="GGGGGGGG", **extra):
    path_ref = write_chains(tmp_path / "ref.pdb", {"A": BINDER, "B": TARGET_B})
    designs = load_designs(pd.DataFrame({"design_name": ["d1"], "seq_binder": ["ACDEFGHI"], "seq_target": [seq_target],
                                         "path_structure_ref": [path_ref], **{k: [v] for k, v in extra.items()}}))
    tall = pd.DataFrame({"design_name": "d1", "model": "of3", "seed": [1], "iptm": [0.8],
                         "path_structure": [write_chains(tmp_path / "pred.pdb", pred_chains)]})
    return designs, tall


def test_add_structure_rmsd_fills_columns(tmp_path):
    designs, tall = designs_and_tall(tmp_path, {"A": BINDER + [3, 0, 0], "B": TARGET_B})
    row = add_structure_rmsd(tall, designs).iloc[0]
    assert row.rmsd_fold == pytest.approx(0, abs=1e-3) and row.rmsd_pose_target_B == pytest.approx(3, abs=1e-3)


def test_swapped_identical_target_copies_are_matched(tmp_path):
    path_ref = write_chains(tmp_path / "ref.pdb", {"A": BINDER, "B": TARGET_B, "C": TARGET_C})
    designs = load_designs(pd.DataFrame({"design_name": ["d1"], "seq_binder": ["ACDEFGHI"], "seq_target": ["GGGGGGGG,GGGGGGGG"],
                                         "path_structure_ref": [path_ref]}))
    tall = pd.DataFrame({"design_name": "d1", "path_structure": [write_chains(tmp_path / "pred.pdb", {"A": BINDER, "B": TARGET_C, "C": TARGET_B})]})
    row = add_structure_rmsd(tall, designs).iloc[0]
    assert row.rmsd_pose_target_B == pytest.approx(0, abs=1e-3) and row.rmsd_pose_target_C == pytest.approx(0, abs=1e-3)

    # Different target sequences: no swapping, so the pose RMSDs against the mismatched positions are large.
    ref_seqs, pred_seqs = {"C": "A" * 8}, {"C": "A" * 8}
    designs = designs.assign(seq_target="GGGGGGGG,AAAAAAAA", path_structure_ref=write_chains(tmp_path / "ref2.pdb", {"A": BINDER, "B": TARGET_B, "C": TARGET_C}, ref_seqs))
    tall = tall.assign(path_structure=write_chains(tmp_path / "pred2.pdb", {"A": BINDER, "B": TARGET_C, "C": TARGET_B}, pred_seqs))
    assert add_structure_rmsd(tall, designs).iloc[0].rmsd_pose_target_B > 1


def test_no_reference_or_unusable_files_give_blanks_not_errors(tmp_path, capsys):
    designs, tall = designs_and_tall(tmp_path, {"A": BINDER, "B": TARGET_B})
    assert "rmsd_fold" not in add_structure_rmsd(tall, designs.assign(path_structure_ref=""))
    missing = add_structure_rmsd(tall, designs.assign(path_structure_ref=str(tmp_path / "nope.pdb")))
    assert "rmsd_fold" not in missing and "WARNING: no RMSD for d1" in capsys.readouterr().out
    short = add_structure_rmsd(tall.assign(path_structure=write_chains(tmp_path / "short.pdb", {"A": BINDER, "B": TARGET_B[:5]})), designs)
    assert "rmsd_fold" not in short and "residues" in capsys.readouterr().out


def test_refiner_structure_path_becomes_the_reference_and_chain_ids_are_checked():
    refiner_csv = pd.DataFrame({"design_name": ["d1"], "seq_binder": ["ACDE"], "seq_target": ["GG"], "path_structure": ["/outputs/x.cif"]})
    assert load_designs(refiner_csv).path_structure_ref.iloc[0] == "/outputs/x.cif"
    assert load_designs(refiner_csv.assign(ref_chain_ids="H,L")).ref_chain_ids.iloc[0] == "H,L"
    with pytest.raises(ValueError, match=r"line 2 \(d1\): ref_chain_ids"):
        load_designs(refiner_csv.assign(ref_chain_ids="H,L,X"))
    with pytest.raises(ValueError, match="ref_chain_ids"):
        load_designs(refiner_csv.assign(ref_chain_ids="H,H"))


def test_summary_reports_each_target_and_the_worst_seed_averaged_one():
    designs = load_designs(pd.DataFrame({"design_name": ["d1"], "seq_binder": ["AC"], "seq_target": ["GG,GG"]}))
    tall = pd.DataFrame({"design_name": "d1", "model": "of3", "seed": [1, 2], "rmsd_fold": [1.0, 3.0],
                         "rmsd_pose_target_B": [2.0, 4.0], "rmsd_pose_target_C": [1.0, 9.0]})
    summary = summarize(tall, designs).iloc[0]
    assert summary.rmsd_fold_mean == 2 and summary.rmsd_pose_target_B_mean == 3 and summary.rmsd_pose_target_C_max == 9
    assert summary.rmsd_pose_worst_target_mean == 5      # target C's mean (1 + 9) / 2; the mean of each seed's worst would be (2 + 9) / 2


def test_summary_without_rmsd_columns_has_no_worst_target():
    designs = load_designs(pd.DataFrame({"design_name": ["d1"], "seq_binder": ["AC"], "seq_target": ["GG"]}))
    tall = pd.DataFrame({"design_name": "d1", "model": "of3", "seed": [1], "iptm": [0.5]})
    assert "rmsd_pose_worst_target_mean" not in summarize(tall, designs)


def test_converter_records_reference_file_and_chains(tmp_path):
    write_chains(tmp_path / "d1.pdb", {"H": BINDER, "L": TARGET_B, "X": TARGET_C})
    row = designs_from_structures(tmp_path, binder_chain="H").iloc[0]
    assert row.path_structure_ref == str((tmp_path / "d1.pdb").resolve()) and row.ref_chain_ids == "H,L,X"


# ------------------------------------------------------------------------------------- sequence safety net

def test_matching_sequences_pass_the_check(tmp_path):
    ref = {"H": BINDER, "L": TARGET_B}
    out = rmsds(tmp_path, ref, {"A": BINDER, "B": TARGET_B}, ["H", "L"], ["A", "B"], [BINDER_SEQ, TARGET_SEQ])
    assert out["rmsd_fold"] == pytest.approx(0, abs=1e-3)


def test_same_length_but_wrong_chain_is_caught_in_the_reference(tmp_path):
    # Blank ref_chain_ids would pick chain A, which here is a different protein of the same length.
    with pytest.raises(ValueError, match="reference chain A is not the binder sequence .first at position 1: expected A, found W"):
        rmsds(tmp_path, {"A": BINDER, "B": TARGET_B}, {"A": BINDER, "B": TARGET_B}, ["A", "B"], ["A", "B"],
              [BINDER_SEQ, TARGET_SEQ], ref_seqs={"A": "WWWWWWWW"})


def test_wrong_target_chain_is_caught_in_the_prediction(tmp_path):
    with pytest.raises(ValueError, match="predicted chain B is not the target B sequence"):
        rmsds(tmp_path, {"A": BINDER, "B": TARGET_B}, {"A": BINDER, "B": TARGET_B}, ["A", "B"], ["A", "B"],
              [BINDER_SEQ, TARGET_SEQ], pred_seqs={"B": "AGGGGGGG"})


def test_sequence_mismatch_in_the_table_gives_blanks_and_a_warning(tmp_path, capsys):
    designs, tall = designs_and_tall(tmp_path, {"A": BINDER, "B": TARGET_B})
    designs = designs.assign(seq_binder="WWWWWWWW")
    assert "rmsd_fold" not in add_structure_rmsd(tall, designs)
    assert "is not the binder sequence" in capsys.readouterr().out


# ------------------------------------------------------------------------------ staging and the executor

def test_reference_files_are_staged_for_upload_and_volume_paths_are_kept(tmp_path):
    local = write_chains(tmp_path / "mine.pdb", {"A": BINDER, "B": TARGET_B})
    designs = load_designs(pd.DataFrame({"design_name": ["d1", "d2", "d3"], "seq_binder": ["ACDEFGHI"] * 3, "seq_target": ["GGGGGGGG"] * 3,
                                         "path_structure_ref": [local, "/outputs/run/runs/run_0/x_cycle_1.cif", ""]}))
    staged, uploads = plan_reference_uploads(designs, "my_val", "/outputs")
    assert uploads == [(local, "my_val/reference/d1.pdb")]
    assert staged.path_structure_ref.tolist() == ["/outputs/my_val/reference/d1.pdb", "/outputs/run/runs/run_0/x_cycle_1.cif", ""]
    no_column = load_designs(designs.drop(columns="path_structure_ref"))
    assert plan_reference_uploads(no_column, "my_val", "/outputs")[1] == []


def test_unknown_reference_paths_fail_before_anything_starts(tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("x")
    designs = load_designs(pd.DataFrame({"design_name": ["d1", "d2"], "seq_binder": ["ACDEFGHI"] * 2, "seq_target": ["GGGGGGGG"] * 2,
                                         "path_structure_ref": [str(tmp_path / "gone.pdb"), str(notes)]}))
    with pytest.raises(ValueError) as error:
        plan_reference_uploads(designs, "out", "/outputs")
    text = str(error.value)
    assert "line 2 (d1)" in text and "neither a file on this machine" in text and "line 3 (d2)" in text and ".pdb or .cif" in text


class Done:
    def __init__(self, result):
        self.result = result

    def get(self, timeout=0):
        return self.result


def test_executor_adds_rmsd_columns_and_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(modal_common, "OUTPUTS_MOUNT", str(tmp_path))
    monkeypatch.setattr(modal_common.outputs_volume, "commit", lambda: None)
    reloads = []
    monkeypatch.setattr(modal_common.outputs_volume, "reload", lambda: reloads.append(1))
    path_ref = write_chains(tmp_path / "ref.pdb", {"A": BINDER, "B": TARGET_B})
    designs = load_designs(pd.DataFrame({"design_name": ["d1"], "seq_binder": [BINDER_SEQ], "seq_target": [TARGET_SEQ],
                                         "path_structure_ref": [path_ref]}))

    def spawn(job, path_job, gpu_type):
        path_pred = write_chains(tmp_path / f"pred{job['seed']}.pdb", {"A": BINDER + [job["seed"], 0, 0], "B": TARGET_B})
        return Done([{"iptm": 0.8, "path_structure": path_pred, "paratope_indices_B": "1", "epitope_indices_B": "1"}])

    monkeypatch.setattr(modal_common, "_spawn_validation_job", spawn)
    modal_common.execute_validation(designs, ["of3"], [2, 4], 1, "out", poll_seconds=0, flush_every=1)
    tall = pd.read_csv(tmp_path / "out" / "validation_results.csv")
    assert tall.rmsd_fold.abs().max() < 1e-3 and sorted(tall.rmsd_pose_target_B.round(2)) == [2, 4] and reloads
    summary = pd.read_csv(tmp_path / "out" / "validation_summary.csv").iloc[0]
    assert summary.rmsd_pose_target_B_mean == pytest.approx(3, abs=1e-2) and summary.rmsd_pose_worst_target_mean == pytest.approx(3, abs=1e-2)


def test_a_failing_rmsd_step_does_not_cost_the_run_its_results(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(modal_common, "OUTPUTS_MOUNT", str(tmp_path))
    monkeypatch.setattr(modal_common.outputs_volume, "commit", lambda: None)
    monkeypatch.setattr(modal_common.outputs_volume, "reload", lambda: (_ for _ in ()).throw(RuntimeError("volume busy")))
    path_ref = write_chains(tmp_path / "ref.pdb", {"A": BINDER, "B": TARGET_B})
    designs = load_designs(pd.DataFrame({"design_name": ["d1"], "seq_binder": [BINDER_SEQ], "seq_target": [TARGET_SEQ],
                                         "path_structure_ref": [path_ref]}))
    monkeypatch.setattr(modal_common, "_spawn_validation_job",
                        lambda job, path_job, gpu_type: Done([{"iptm": 0.8, "path_structure": path_ref, "paratope_indices_B": "1", "epitope_indices_B": "1"}]))
    modal_common.execute_validation(designs, ["of3"], [1], 1, "out", poll_seconds=0, flush_every=1)
    tall = pd.read_csv(tmp_path / "out" / "validation_results.csv")
    assert len(tall) == 1 and "rmsd_fold" not in tall
    assert "WARNING: structure scoring (RMSD, binder confidence) skipped for 1 job(s): RuntimeError: volume busy" in capsys.readouterr().out


# ------------------------------------------------------------------------------------ binder confidence

def write_with_plddt(path, binder_plddt, target_plddt=(95.0,) * 8):
    atoms = [struc.Atom(xyz, chain_id=chain, res_id=i + 1, res_name="ALA", atom_name="CA", element="C")
             for chain, coords in (("A", BINDER), ("B", TARGET_B)) for i, xyz in enumerate(coords)]
    array = struc.array(atoms)
    array.set_annotation("b_factor", np.array([*binder_plddt, *target_plddt], dtype=float))
    file = pdb.PDBFile()
    file.set_structure(array)
    file.write(str(path))
    return str(path)


def test_binder_plddt70_fraction_counts_only_confident_binder_residues(tmp_path):
    low_target = write_with_plddt(tmp_path / "a.pdb", [90, 80, 70, 69.9, 50, 20, 95, 99], target_plddt=(10.0,) * 8)
    scaled = write_with_plddt(tmp_path / "b.pdb", [0.9, 0.8, 0.7, 0.69, 0.5, 0.2, 0.95, 0.99])    # 0-1 scale
    tall = pd.DataFrame({"path_structure": [low_target, scaled, str(tmp_path / "missing.pdb")]})
    out = add_binder_plddt70_fraction(tall).binder_plddt70_fraction
    assert out[0] == pytest.approx(5 / 8) and out[1] == pytest.approx(5 / 8) and np.isnan(out[2])   # target ignored, bad file blank


def test_binder_confidence_needs_no_reference_and_reaches_the_summary():
    designs = load_designs(pd.DataFrame({"design_name": ["d1"], "seq_binder": ["AC"], "seq_target": ["GG"]}))
    tall = pd.DataFrame({"design_name": "d1", "model": "of3", "seed": [1, 2], "binder_plddt70_fraction": [0.2, 0.6]})
    summary = summarize(tall, designs).iloc[0]
    assert summary.binder_plddt70_fraction_mean == pytest.approx(0.4) and summary.binder_plddt70_fraction_max == 0.6
