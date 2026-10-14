"""
Build a validation input CSV from a folder of predicted/designed complexes (.pdb/.cif).

    python designs_from_structures.py --input_dir structures/ --out_csv designs.csv [--binder_chain A]

Each file is one design: the binder chain becomes `seq_binder`, every other protein chain a target
(`seq_target`, comma-separated, labelled B, C, ... in order, as the pipeline does), and non-polymer
residues become `ligands` (SMILES). The design's own paratope/epitope is written as sequence-position
indices (`ref_paratope_indices_B`, `ref_epitope_indices_B`, ...) so validation can score whether another
model reproduces the same binding mode.
"""
import argparse
from collections import Counter
from pathlib import Path

import biotite.structure as struc
import biotite.structure.info as info
import numpy as np
import pandas as pd

from StrucTools import extract_atom_array, interface_positions

# Monoatomic ions by element -> SMILES. Others (Fe, Cu, Co, ...) have several common charges: pass --ligands.
ION_SMILES = {"MG": "[Mg+2]", "ZN": "[Zn+2]", "MN": "[Mn+2]", "CA": "[Ca+2]", "NA": "[Na+]", "K": "[K+]", "CL": "[Cl-]"}
# Crystallisation additives that are almost never the intended ligand.
ADDITIVES = {"GOL", "EDO", "PEG", "PG4", "PGE", "1PE", "SO4", "PO4", "ACT", "DMS", "MPD", "FMT", "TRS", "EPE", "BME", "IMD"}


def ccd_to_smiles(code: str) -> str:
    """SMILES for a Chemical Component Dictionary code (stereochemistry is not preserved)."""
    from rdkit import Chem  # lazy: only needed when a ligand has to be resolved
    from biotite.interface.rdkit import to_mol
    return Chem.MolToSmiles(Chem.RemoveHs(to_mol(info.residue(code))))


def resolve_ligand(residue) -> tuple[str, str]:
    """(smiles, method) for one non-polymer residue; smiles is '' when it cannot be resolved."""
    if residue.array_length() == 1:
        element = str(residue.element[0]).upper()
        return (ION_SMILES[element], "ion table") if element in ION_SMILES else ("", f"ambiguous ion {element}")
    try:
        return ccd_to_smiles(str(residue.res_name[0])), "CCD"
    except Exception:
        return "", "not a CCD code"


def design_from_structure(path: Path, binder_chain: str, target_chains, exclude_het: set, warn) -> dict:
    atoms = extract_atom_array(str(path))
    protein = atoms[struc.filter_amino_acids(atoms)]
    chains = list(dict.fromkeys(protein.chain_id.tolist()))
    if binder_chain not in chains:
        raise ValueError(f"binder chain {binder_chain} not found (protein chains: {chains})")
    targets = target_chains or [c for c in chains if c != binder_chain]
    if not targets:
        raise ValueError("no target protein chain")

    def sequence(chain):
        chain_atoms = protein[protein.chain_id == chain]
        starts = struc.get_residue_starts(chain_atoms)
        if (np.diff(chain_atoms.res_id[starts]) > 1).any():
            warn(f"{path.name}: chain {chain} has gaps in its residue numbering (kept as one sequence)")
        if chain_atoms.hetero.any():
            modified = sorted({str(n) for n in chain_atoms.res_name[chain_atoms.hetero]})
            warn(f"{path.name}: chain {chain} has modified residues {modified}, read as their parent amino acid")
        seq = "".join(str(s) for s in struc.to_sequence(chain_atoms)[0])
        assert len(seq) == len(starts), f"{path.name}: chain {chain} sequence/residue count mismatch"
        return seq

    row = {"design_name": path.stem, "seq_binder": sequence(binder_chain),
           "seq_target": ",".join(sequence(c) for c in targets)}

    het = atoms[atoms.hetero & ~struc.filter_amino_acids(atoms) & ~struc.filter_solvent(atoms)]
    ligands = []
    for start, stop in zip(struc.get_residue_starts(het), list(struc.get_residue_starts(het))[1:] + [het.array_length()]):
        residue = het[start:stop]
        if str(residue.res_name[0]) in ADDITIVES | exclude_het:
            continue
        ligands.append((str(residue.res_name[0]), str(residue.chain_id[0]), *resolve_ligand(residue)))
    row["ligands"] = ",".join(smiles for *_, smiles, _ in ligands if smiles)
    row["_ligands"] = ligands

    for letter, chain in zip("BCDEFGHIJ", targets):
        paratope, epitope = interface_positions(atoms, binder_chain, chain)
        row[f"ref_paratope_indices_{letter}"] = ",".join(map(str, paratope))
        row[f"ref_epitope_indices_{letter}"] = ",".join(map(str, epitope))
    if not any(row[f"ref_paratope_indices_{l}"] for l in "BCDEFGHIJ"[:len(targets)]):
        warn(f"{path.name}: binder makes no contact with any target chain")
    return row


def designs_from_structures(input_dir, binder_chain="A", target_chains=None, ligands=None,
                            exclude_het=(), strict=False) -> pd.DataFrame:
    """One row per structure file in `input_dir`. `ligands` (SMILES list) replaces the extracted ligands."""
    warnings = []
    rows = []
    for path in sorted(Path(input_dir).glob("*")):
        if path.suffix.lower() in (".pdb", ".cif"):
            rows.append(design_from_structure(path, binder_chain, target_chains, set(exclude_het), warnings.append))
    if not rows:
        raise ValueError(f"no .pdb/.cif files in {input_dir}")

    summary = Counter((code, smiles, method) for row in rows for code, _, smiles, method in row["_ligands"])
    unresolved = [k for k in summary if not k[1]]
    if ligands is not None:
        print(f"Ligands overridden for every design: {ligands}")
    elif summary:
        print("Ligands found (count over all designs):")
        for (code, smiles, method), n in summary.items():
            print(f"  {n:4d} x {code:5s} {smiles or 'UNRESOLVED':30s} [{method}]")
        if unresolved and strict:
            raise ValueError(f"unresolved ligands {[k[0] for k in unresolved]}: pass --ligands or --exclude_het")
        if unresolved:
            print("  UNRESOLVED ligands are left out of the CSV: pass --ligands or --exclude_het, or use --strict to fail.")
    for message in dict.fromkeys(warnings):
        print("WARNING:", message)

    df = pd.DataFrame(rows).drop(columns="_ligands")
    if ligands is not None:
        df["ligands"] = ",".join(ligands)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--binder_chain", default="A")
    parser.add_argument("--target_chains", default="", help="comma-separated; default: every other protein chain")
    parser.add_argument("--ligands", default=None, help="comma-separated SMILES used for every design instead of the extracted ones")
    parser.add_argument("--exclude_het", default="", help="comma-separated residue names to ignore, e.g. 'HEM,NAG'")
    parser.add_argument("--strict", action="store_true", help="fail if any ligand cannot be resolved to SMILES")
    args = parser.parse_args()
    df = designs_from_structures(
        args.input_dir, args.binder_chain, [c for c in args.target_chains.split(",") if c] or None,
        None if args.ligands is None else [s for s in args.ligands.split(",") if s],
        [c for c in args.exclude_het.split(",") if c], args.strict)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(df)} designs to {args.out_csv}")
