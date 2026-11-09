import numpy as np
import pandas as pd
import pytest

from validation_core import (add_contact_check, add_hotspot_recall, add_interface_consistency, check_hotspots, drop_coverage_columns,
                             expand_jobs, interface_consistency, job_folder, load_designs, summarize)

DESIGNS = pd.DataFrame({"design_name": ["d1", "d2"], "seq_binder": ["ACDE", "ACDF"], "seq_target": ["GGHH", "GGHH,GGHH"]})


def test_load_designs_ok_and_refiner_columns_become_reference():
    df = DESIGNS.assign(paratope_indices_B="1,2", epitope_indices_B="3")
    out = load_designs(df)
    assert "ref_paratope_indices_B" in out and "paratope_indices_B" not in out


def test_load_designs_reports_every_problem_with_line_numbers():
    bad = pd.DataFrame({"design_name": ["a/b", "x", "x"], "seq_binder": ["ACD", "ACZ", ""], "seq_target": ["AAA"] * 3})
    with pytest.raises(ValueError) as error:
        load_designs(bad)
    text = str(error.value)
    assert "line 2" in text and "line 3" in text and "line 4" in text and "repeated" in text


def test_load_designs_missing_column():
    with pytest.raises(ValueError, match="seq_target"):
        load_designs(DESIGNS.drop(columns="seq_target"))


def test_expand_jobs_and_guard():
    jobs = expand_jobs(DESIGNS, ["of3", "esmfold2"], [1, 2])
    assert len(jobs) == 8 and jobs[0] == {**jobs[0], "design_name": "d1", "model": "of3", "seed": 1}
    with pytest.raises(ValueError, match="max_jobs"):
        expand_jobs(DESIGNS, ["of3", "esmfold2"], [1, 2], max_jobs=5)
    with pytest.raises(ValueError, match="models"):
        expand_jobs(DESIGNS, ["esmfold2_fast"], [1])
    with pytest.raises(ValueError, match="num_samples"):
        expand_jobs(DESIGNS, ["of3"], [1], num_samples=0)
    assert expand_jobs(DESIGNS, ["of3"], [1], num_samples=3)[0]["num_samples"] == 3
    assert job_folder("d1", "of3", 21) == "d1/of3_seed21"


def test_consistency_identical_and_partial():
    ref = {"paratope_indices_B": "1,2,3,4", "epitope_indices_B": "10,11"}
    same = interface_consistency(ref, ref, "GGHH")
    assert same["paratope_recall"] == 1 and same["epitope_jaccard"] == 1 and same["chain_assignment"] == "identity"
    part = interface_consistency(ref, {"paratope_indices_B": "1,2,9,8", "epitope_indices_B": "10,11"}, "GGHH")
    assert part["paratope_recall"] == 0.5 and part["paratope_jaccard"] == pytest.approx(2 / 6)


def test_consistency_finds_swapped_identical_chains_but_not_different_ones():
    ref = {"paratope_indices_B": "1,2", "epitope_indices_B": "10", "paratope_indices_C": "5,6", "epitope_indices_C": "20"}
    swapped = {"paratope_indices_B": "5,6", "epitope_indices_B": "20", "paratope_indices_C": "1,2", "epitope_indices_C": "10"}
    assert interface_consistency(ref, swapped, "GG,GG")["paratope_recall"] == 1
    assert interface_consistency(ref, swapped, "GG,GG")["chain_assignment"] == "B->C;C->B"
    assert interface_consistency(ref, swapped, "GG,HH")["paratope_recall"] == 0   # different targets: no swapping


def test_consistency_without_reference_is_empty():
    assert interface_consistency({}, {"paratope_indices_B": "1"}, "GG") == {}


def test_summary_aggregates_seeds_and_noise_floor():
    designs = load_designs(DESIGNS[:1].assign(ref_paratope_indices_B="1,2", ref_epitope_indices_B="3,4"))
    tall = pd.DataFrame({"design_name": "d1", "model": "of3", "seed": [1, 2], "iptm": [0.6, 0.8],
                         "contact_check_passed": [True, False],
                         "paratope_indices_B": ["1,2", "1,9"], "epitope_indices_B": ["3,4", "3,4"]})
    summary = summarize(add_interface_consistency(tall, designs), designs).iloc[0]
    assert summary.n_seeds == 2 and summary.iptm_mean == pytest.approx(0.7) and summary.iptm_max == 0.8
    assert summary.contact_pass_fraction == 0.5
    assert summary.paratope_recall_mean == pytest.approx(0.75)
    assert summary.seed_paratope_jaccard == pytest.approx(1 / 3) and summary.seed_epitope_jaccard == 1


def test_summary_without_reference_has_no_consistency_columns():
    designs = load_designs(DESIGNS[:1])
    tall = pd.DataFrame({"design_name": "d1", "model": "of3", "seed": [1], "iptm": [0.6]})
    with_consistency = add_interface_consistency(tall, designs)
    assert with_consistency.columns.tolist() == tall.columns.tolist()          # nothing added without a reference
    summary = summarize(with_consistency, designs)
    assert not any("recall" in c or "jaccard" in c for c in summary.columns)


def test_contact_check_from_predicted_indices():
    designs = load_designs(DESIGNS[:1])
    tall = pd.DataFrame({"design_name": "d1", "paratope_indices_B": ["1,2", "1,9"], "epitope_indices_B": ["3,4", "3,4"]})
    assert add_contact_check(tall, designs).contact_check_passed.tolist() == [True, True]
    assert add_contact_check(tall, designs, epitope_residues="B3,B4").contact_check_passed.tolist() == [True, True]
    assert add_contact_check(tall, designs, paratope_residues="A1,A2").contact_check_passed.tolist() == [True, False]


# ------------------------------------------------------------------------------------------ hotspot recall

def test_hotspot_recall_is_the_fraction_of_desired_residues_kept():
    designs = load_designs(DESIGNS[:1])
    tall = pd.DataFrame({"design_name": "d1", "paratope_indices_B": ["1,2,3", "1,9"], "epitope_indices_B": ["3,4,5", "3"]})
    out = add_hotspot_recall(tall, designs, epitope_residues="B3,B4,B8,B9", paratope_residues="A1,A2,A3,A4")
    assert out.hotspot_epitope_recall.tolist() == [0.5, 0.25]
    assert out.hotspot_paratope_recall.tolist() == [0.75, 0.25]


def test_hotspot_recall_only_adds_columns_for_lists_that_are_set():
    designs = load_designs(DESIGNS[:1])
    tall = pd.DataFrame({"design_name": "d1", "paratope_indices_B": ["1,2"], "epitope_indices_B": ["3"]})
    assert add_hotspot_recall(tall, designs).columns.tolist() == tall.columns.tolist()
    only_paratope = add_hotspot_recall(tall, designs, paratope_residues="A1,A2")
    assert "hotspot_paratope_recall" in only_paratope and "hotspot_epitope_recall" not in only_paratope


def test_hotspot_recall_pools_targets_and_unions_the_paratope():
    designs = load_designs(DESIGNS[1:])                      # two target chains, B and C
    tall = pd.DataFrame({"design_name": "d2", "paratope_indices_B": ["1,2"], "epitope_indices_B": ["10"],
                         "paratope_indices_C": ["3"], "epitope_indices_C": ["20"]})
    out = add_hotspot_recall(tall, designs, epitope_residues="B10,C20,C21", paratope_residues="A1,A3")
    assert out.hotspot_epitope_recall.iloc[0] == pytest.approx(2 / 3)
    assert out.hotspot_paratope_recall.iloc[0] == 1          # A1 from chain B's contacts, A3 from chain C's


def test_hotspot_recall_tolerates_missing_indices():
    designs = load_designs(DESIGNS[:1])
    tall = pd.DataFrame({"design_name": "d1", "iptm": [0.5]})
    assert add_hotspot_recall(tall, designs, epitope_residues="B3").hotspot_epitope_recall.tolist() == [0]


def test_hotspot_recall_reaches_the_summary():
    designs = load_designs(DESIGNS[:1])
    tall = pd.DataFrame({"design_name": "d1", "model": "of3", "seed": [1, 2], "paratope_indices_B": ["1,2", "1"], "epitope_indices_B": ["3", "3"]})
    summary = summarize(add_hotspot_recall(tall, designs, paratope_residues="A1,A2"), designs).iloc[0]
    assert (summary.hotspot_paratope_recall_mean, summary.hotspot_paratope_recall_min, summary.hotspot_paratope_recall_max) == (0.75, 0.5, 1)
    assert "hotspot_epitope_recall_mean" not in summary


def test_coverage_columns_are_dropped():
    tall = pd.DataFrame({"iptm": [0.5], "epitope_coverage_recall_B": [0], "epitope_coverage_f1_B": [0], "epitope_indices_B": ["3"]})
    assert drop_coverage_columns(tall).columns.tolist() == ["iptm", "epitope_indices_B"]


# ---------------------------------------------------------------------------------------- launch-time check

def test_check_hotspots_accepts_valid_lists_and_blanks():
    designs = load_designs(DESIGNS)
    check_hotspots(designs)
    check_hotspots(designs, "B1,B2", "A1,A4")
    check_hotspots(designs[1:], "B1,C3", "A1,A4")             # chain C exists only on the two-target design


def test_check_hotspots_rejects_bare_numbers():
    with pytest.raises(ValueError, match="'12' must look like"):
        check_hotspots(load_designs(DESIGNS), epitope_residues="12,13")
    with pytest.raises(ValueError, match="'26' must look like"):
        check_hotspots(load_designs(DESIGNS), paratope_residues="26")


def test_check_hotspots_rejects_wrong_chains_and_out_of_range_positions():
    designs = load_designs(DESIGNS)                           # d1: binder 4, one target of 4; d2: binder 4, two targets
    with pytest.raises(ValueError) as error:
        check_hotspots(designs, epitope_residues="C1,B9", paratope_residues="B1,A5")
    text = str(error.value)
    assert "'C1' names chain C" in text and "d1" in text      # d1 has no chain C
    assert "'B9' is outside the 4-residue chain B" in text
    assert "'B1' names chain B, but this design's binder is chain A" in text
    assert "'A5' is outside the 4-residue chain A" in text


def test_check_hotspots_rejects_non_string():
    with pytest.raises(ValueError, match="comma-separated string"):
        check_hotspots(load_designs(DESIGNS), epitope_residues=["B1"])
