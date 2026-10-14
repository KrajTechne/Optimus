import pandas as pd
import pytest

import modal_common
import validation_core as vc


class FakeCall:
    """Stands in for a Modal FunctionCall: not ready for `waits` polls, then returns samples or raises."""
    def __init__(self, result, waits=0):
        self.result, self.waits = result, waits

    def get(self, timeout=0):
        if self.waits > 0:
            self.waits -= 1
            raise TimeoutError
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def samples(iptm):
    return [{"iptm": iptm - 0.1, "paratope_indices_B": "9"}, {"iptm": iptm, "paratope_indices_B": "1,2", "epitope_indices_B": "5,6"},
            {"iptm": 0.1, "chain_ids": ["A", "B"]}]


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(modal_common, "OUTPUTS_MOUNT", str(tmp_path))
    monkeypatch.setattr(modal_common.outputs_volume, "commit", lambda: None)
    designs = vc.load_designs(pd.DataFrame({"design_name": ["good", "bad"], "seq_binder": ["ACDE", "ACDF"], "seq_target": ["GGHH", "GGHH"],
                                            "ref_paratope_indices_B": ["1,2", "1,2"], "ref_epitope_indices_B": ["5,6", "5,6"]}))

    def execute(spawn):
        monkeypatch.setattr(modal_common, "_spawn_validation_job", spawn)
        modal_common.execute_validation(designs, ["of3"], [1, 2], 3, "out", paratope_residues="A1,A2", poll_seconds=0, flush_every=2)
        return tmp_path / "out"
    return execute


def test_results_summary_and_failures_are_written(run):
    def spawn(job, path_job, gpu_type):
        if job["design_name"] == "bad" and job["seed"] == 2:
            return FakeCall(RuntimeError("out of memory"))
        return FakeCall(samples(0.8), waits=1 if job["seed"] == 1 else 0)

    out = run(spawn)
    tall = pd.read_csv(out / "validation_results.csv")
    assert len(tall) == 3 and set(tall.iptm) == {0.8}                    # best sample by iptm; list-valued metrics dropped
    assert "chain_ids" not in tall and tall.contact_check_passed.all()
    assert tall.path_job.str.contains("of3_seed").all() and (tall.paratope_recall == 1).all()
    failures = pd.read_csv(out / "failures.csv")
    assert failures.design_name.tolist() == ["bad"] and "out of memory" in failures.error[0]
    summary = pd.read_csv(out / "validation_summary.csv")
    assert summary.set_index("design_name").n_seeds.to_dict() == {"good": 2, "bad": 1}


def test_job_folder_is_passed_to_the_runner(run, tmp_path):
    seen = []
    run(lambda job, path_job, gpu_type: seen.append(path_job) or FakeCall(samples(0.5)))
    assert sorted(p.replace("\\", "/") for p in seen)[0] == f"{tmp_path}/out/bad/of3_seed1".replace("\\", "/")
