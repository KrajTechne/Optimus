from StrucTools import binder_binds_all_targets, binder_binds_contacts

# Interface as determine_binding_interface() reports it: comma-separated residue numbers per target chain.
METRICS = {
    "paratope_indices_B": "10,11,12", "epitope_indices_B": "73,74,75",
    "paratope_indices_C": "40,41",    "epitope_indices_C": "73,80",
}


def test_no_constraints_always_passes():
    assert binder_binds_contacts(METRICS, "B")
    assert binder_binds_contacts({"paratope_indices_B": "", "epitope_indices_B": ""}, "B")


def test_epitope_needs_two_contacted_residues():
    assert binder_binds_contacts(METRICS, "B", epitope_residues="B73,B74,B99")
    assert not binder_binds_contacts(METRICS, "B", epitope_residues="B73,B98,B99")


def test_epitope_residues_are_filtered_to_the_chain():
    # Only chain C is constrained, so chain B has nothing to check even though its epitope is unrelated.
    assert binder_binds_contacts(METRICS, "B", epitope_residues="C73,C80")
    assert binder_binds_contacts(METRICS, "C", epitope_residues="C73,C80")


def test_paratope_needs_two_contacted_residues():
    assert binder_binds_contacts(METRICS, "B", paratope_residues="A10,A12")
    assert not binder_binds_contacts(METRICS, "B", paratope_residues="A10,A99")


def test_all_targets_must_pass():
    assert binder_binds_all_targets(METRICS, 2, epitope_residues="B73,B74,C73,C80")
    assert not binder_binds_all_targets(METRICS, 2, epitope_residues="B73,B74,C73,C99")
    assert binder_binds_all_targets(METRICS, 1, epitope_residues="B73,B74,C73,C99")  # chain C is not a target here
