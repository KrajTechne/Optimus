import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import numpy as np
import pytest

from designs_from_structures import designs_from_structures
from StrucTools import determine_binding_interface, interface_positions


def residue(chain, res_id, name, xyz, hetero=False, atom="CA", element="C"):
    atom_array = struc.array([struc.Atom(xyz, chain_id=chain, res_id=res_id, res_name=name, atom_name=atom,
                                         element=element, hetero=hetero)])
    return atom_array


def write_complex(path, binder, target, extra=()):
    """binder/target: list of (res_id, res_name, x, hetero); residues sit on a line along x (CA only)."""
    parts = [residue("A", i, n, [x, 0, 0], h) for i, n, x, h in binder]
    parts += [residue("B", i, n, [x, 0, 0], h) for i, n, x, h in target]
    parts += list(extra)
    atoms = sum(parts[1:], parts[0])
    file = pdb.PDBFile()
    file.set_structure(atoms)
    file.write(str(path))


@pytest.fixture
def folder(tmp_path):
    # Binder residues 100-102 (numbering offset), target 5,6,8 (gap) + modified SEP; Mg ion with a placeholder name.
    binder = [(100, "ALA", 0.0, False), (101, "GLY", 3.8, False), (102, "LEU", 100.0, False)]
    target = [(5, "SER", 2.0, False), (6, "SEP", 5.0, True), (8, "VAL", 50.0, False)]
    ion = residue("C", 1, "l01", [20.0, 20.0, 20.0], hetero=True, atom="MG1", element="MG")
    write_complex(tmp_path / "d1.pdb", binder, target, [ion])
    return tmp_path


def test_sequences_ligand_and_interface(folder):
    row = designs_from_structures(folder).iloc[0]
    assert row.seq_binder == "AGL"
    assert row.seq_target == "SSV"                         # SEP read as its parent amino acid
    assert row.ligands == "[Mg+2]"
    assert row.ref_paratope_indices_B == "1,2"             # sequence positions, not 100/101
    assert row.ref_epitope_indices_B == "1,2"              # positions of residues 5 and 6, despite the gap before 8


def test_interface_positions_match_determine_binding_interface(tmp_path):
    binder = [(100, "ALA", 0.0, False), (101, "GLY", 3.8, False), (102, "LEU", 100.0, False)]
    target = [(5, "SER", 2.0, False), (6, "THR", 5.0, False), (8, "VAL", 50.0, False)]
    write_complex(tmp_path / "plain.pdb", binder, target)
    atoms = pdb.PDBFile.read(str(tmp_path / "plain.pdb")).get_structure(model=1)
    assert interface_positions(atoms, "A", "B") == ([1, 2], [1, 2])
    reference = determine_binding_interface(str(tmp_path / "plain.pdb"), [], "A", "B")  # file residue IDs
    assert reference["paratope_indices_B"] == "100,101" and reference["epitope_indices_B"] == "5,6"


def test_ligand_override(folder):
    assert designs_from_structures(folder, ligands=["CCO"]).iloc[0].ligands == "CCO"


def test_unresolvable_ligand_is_left_out_or_fails_strict(tmp_path):
    binder = [(1, "ALA", 0.0, False)]
    target = [(1, "SER", 2.0, False)]
    two_atoms = residue("C", 1, "xyz", [20.0, 20.0, 20.0], hetero=True) + residue("C", 1, "xyz", [21.0, 20.0, 20.0], hetero=True)
    write_complex(tmp_path / "d.pdb", binder, target, [two_atoms])
    assert designs_from_structures(tmp_path).iloc[0].ligands == ""
    with pytest.raises(ValueError, match="unresolved ligands"):
        designs_from_structures(tmp_path, strict=True)


def test_missing_binder_chain(folder):
    with pytest.raises(ValueError, match="binder chain Z"):
        designs_from_structures(folder, binder_chain="Z")
