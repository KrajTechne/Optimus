import numpy as np
import pandas as pd
import pytest

from test_structure_rmsd import BINDER, BINDER_SEQ, write_chains
from validation_core import (_assignments, add_interface_consistency, add_structure_rmsd, check_segments, design_segments,
                             interface_consistency, load_designs, parse_segments, summarize)

# A target that is two identical halves joined by a short linker, in one chain (like a linked homodimer).
HALF = "G" * 8
SEQ_TARGET = HALF + "AA" + HALF
SEGMENTS = {"B": [(1, 8), (11, 18)]}
MOVE = "B:1-8->11-18,11-18->1-8"
H1 = np.random.default_rng(1).normal(scale=8, size=(8, 3)) + [10, 0, 0]
C2 = np.array([-1, -1, 1])                                   # a 180 degree turn about z maps one half onto the other
TARGET = np.vstack([H1, [[0, 0, -1.0], [0, 0, 1.0]], H1 * C2])  # linker atoms sit on the axis


def designs_for(column_value=None, **extra):
    frame = pd.DataFrame({"design_name": ["d1"], "seq_binder": [BINDER_SEQ], "seq_target": [SEQ_TARGET], **extra})
    if column_value is not None:
        frame["identical_segments_target_B"] = column_value
    return load_designs(frame)


# ------------------------------------------------------------------------------------------ the column

def test_segments_parse_and_pass_the_checks():
    assert parse_segments("1-130,136-265") == [(1, 130), (136, 265)]
    check_segments([(1, 8), (11, 18)], SEQ_TARGET)


@pytest.mark.parametrize("text, message", [
    ("1-8", "needs 2 to"), ("1-8,11-19", "same length"), ("1-8,13-20", "runs past the end"), ("1-8,8-15", "overlap"),
    ("1-8,3-10", "overlap"), ("1-8,9-16", "identical sequences"), ("1-8,11", "must look like"), ("0-7,11-18", "must look like"),
])
def test_bad_segments_are_reported_with_the_csv_line(text, message):
    with pytest.raises(ValueError) as error:
        designs_for(text)
    assert "line 2 (d1): identical_segments_target_B" in str(error.value) and message in str(error.value)


def test_segments_for_a_missing_target_chain_are_rejected_and_blank_is_allowed():
    frame = pd.DataFrame({"design_name": ["d1"], "seq_binder": [BINDER_SEQ], "seq_target": [SEQ_TARGET],
                          "identical_segments_target_C": ["1-8,11-18"]})
    with pytest.raises(ValueError, match="no target chain C"):
        load_designs(frame)
    assert design_segments(designs_for("").iloc[0]) == {}
    assert design_segments(designs_for("1-8,11-18").iloc[0]) == SEGMENTS


# ----------------------------------------------------------------------------------------- assignments

def test_assignments_combine_chain_swaps_and_segment_swaps():
    found = list(_assignments("GG,GG", {"B": [(1, 1), (2, 2)]}))
    assert len(found) == 4 and found[0][2] == "identity"                       # 2 chain relabelings x 2 segment orders
    assert {label for *_, label in found} == {"identity", "B:1-1->2-2,2-2->1-1", "B->C;C->B", "B->C;C->B;B:1-1->2-2,2-2->1-1"}
    assert len(list(_assignments("GG,GG"))) == 2 and next(iter(_assignments("GG")))[2] == "identity"


# ------------------------------------------------------------------------------ interface consistency

def test_interface_overlap_follows_the_identical_segments():
    ref = {"paratope_indices_B": "5,6", "epitope_indices_B": "3,4"}
    pred = {"paratope_indices_B": "5,6", "epitope_indices_B": "13,14"}          # binder on the other half
    assert interface_consistency(ref, pred, SEQ_TARGET)["epitope_recall"] == 0
    swapped = interface_consistency(ref, pred, SEQ_TARGET, SEGMENTS)
    assert swapped["epitope_recall"] == 1 and swapped["chain_assignment"] == MOVE
    same = interface_consistency(ref, ref, SEQ_TARGET, SEGMENTS)
    assert same["epitope_jaccard"] == 1 and same["chain_assignment"] == "identity"
    outside = interface_consistency({"epitope_indices_B": "9"}, {"epitope_indices_B": "10"}, SEQ_TARGET, SEGMENTS)
    assert outside["epitope_recall"] == 0                                        # linker residues are not swapped


def test_seed_noise_and_per_job_consistency_use_the_design_column():
    designs = designs_for("1-8,11-18", ref_paratope_indices_B="5", ref_epitope_indices_B="3,4")
    tall = pd.DataFrame({"design_name": "d1", "model": "of3", "seed": [1, 2], "paratope_indices_B": "5",
                         "epitope_indices_B": ["3,4", "13,14"]})
    with_column = add_interface_consistency(tall, designs)
    assert with_column.epitope_recall.tolist() == [1, 1] and with_column.chain_assignment.tolist() == ["identity", MOVE]
    assert summarize(with_column, designs).iloc[0].seed_epitope_jaccard == 1
    plain = designs_for(None, ref_paratope_indices_B="5", ref_epitope_indices_B="3,4")
    assert add_interface_consistency(tall, plain).epitope_recall.tolist() == [1, 0]


# ------------------------------------------------------------------------------------------------ RMSD

def symmetric_case(tmp_path, column_value):
    path_ref = write_chains(tmp_path / "ref.pdb", {"A": BINDER, "B": TARGET}, {"B": SEQ_TARGET})
    # Same target, binder on the other half: the symmetric-equivalent of the reference.
    path_pred = write_chains(tmp_path / "pred.pdb", {"A": BINDER * C2, "B": TARGET}, {"B": SEQ_TARGET})
    designs = designs_for(column_value, path_structure_ref=path_ref)
    return add_structure_rmsd(pd.DataFrame({"design_name": "d1", "path_structure": [path_pred]}), designs).iloc[0]


def test_pose_rmsd_is_measured_both_ways_and_the_lower_one_kept(tmp_path):
    row = symmetric_case(tmp_path, "1-8,11-18")
    assert row.rmsd_fold == pytest.approx(0, abs=1e-3) and row.rmsd_pose_target_B == pytest.approx(0, abs=1e-3)
    assert row.rmsd_assignment == MOVE


def test_without_the_column_the_swapped_binding_looks_like_a_large_pose_error(tmp_path):
    row = symmetric_case(tmp_path, None)
    assert row.rmsd_fold == pytest.approx(0, abs=1e-3) and row.rmsd_pose_target_B > 5 and row.rmsd_assignment == "identity"


def test_column_present_but_binder_on_the_same_half_keeps_the_identity(tmp_path):
    path_ref = write_chains(tmp_path / "ref.pdb", {"A": BINDER, "B": TARGET}, {"B": SEQ_TARGET})
    designs = designs_for("1-8,11-18", path_structure_ref=path_ref)
    row = add_structure_rmsd(pd.DataFrame({"design_name": "d1", "path_structure": [path_ref]}), designs).iloc[0]
    assert row.rmsd_pose_target_B == pytest.approx(0, abs=1e-3) and row.rmsd_assignment == "identity"
