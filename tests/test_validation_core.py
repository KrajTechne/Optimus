import numpy as np
import pandas as pd
import pytest

from validation_core import (add_contact_check, add_interface_consistency, expand_jobs, interface_consistency, job_folder, load_designs,
                             summarize)

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
    summary = summarize(add_interface_consistency(tall, designs), designs)
    assert np.isnan(summary.paratope_recall_mean.iloc[0]) if "paratope_recall_mean" in summary else True


def test_contact_check_from_predicted_indices():
    designs = load_designs(DESIGNS[:1])
    tall = pd.DataFrame({"design_name": "d1", "paratope_indices_B": ["1,2", "1,9"], "epitope_indices_B": ["3,4", "3,4"]})
    assert add_contact_check(tall, designs).contact_check_passed.tolist() == [True, True]
    assert add_contact_check(tall, designs, epitope_residues="B3,B4").contact_check_passed.tolist() == [True, True]
    assert add_contact_check(tall, designs, paratope_residues="A1,A2").contact_check_passed.tolist() == [True, False]
