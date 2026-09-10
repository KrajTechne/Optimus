import argparse
import os
import shutil
import yaml
import pandas as pd
import numpy as np

from RunBoltz2 import RunBoltz2
from RunESMFold2 import RunESMFold2
from LigandMPNN.wrapper import LigandMPNNWrapper
from StrucTools import convert_cif_to_pdb

def load_model_setup_run(
        model_name: str, 
        design_name: str, 
        seq_binder: str, 
        seq_target: str, 
        path_output_dir: str,
        ligands: str = "",
        epitope_residues: str = "",
        **kwargs):
    """ Load the model and pass in the design information to initialize the refinment process.
        Intialize also the MPNN wrapper
        This function returns a class instance of the model that can be used to run the refinement process.
    """
    # Convert inputs into desired format for each Structure Prediction Model
    seq_list = [seq_binder] + seq_target.split(",")
    if ligands == "":
        ligand_list = []
    else:
        ligand_list = ligands.split(",")
    if epitope_residues == "":
        desired_epitope_residues = []
    else:
        desired_epitope_residues = [x for x in epitope_residues.split(",") if x.strip()]

    # Structure Prediction Model Initialization
    if model_name in ['ESMFold2', 'ESMFold2-Fast']:
        model = RunESMFold2(design_name = design_name, model_name = model_name, seq_list = seq_list, 
                            path_output_dir = path_output_dir, ligand_list= ligand_list, 
                            desired_epitope_residues = desired_epitope_residues, **kwargs)
    elif model_name == 'Boltz2':
        model = RunBoltz2(design_name = design_name, seq_list = seq_list, path_output_dir = path_output_dir,
                          ligand_list= ligand_list, desired_epitope_residues= desired_epitope_residues, **kwargs)
    else:
        raise ValueError(f"Model name {model_name} is not supported. Please choose from ['ESMFold2', 'Boltz2', 'ESMFold2-Fast']")

    # MPNN Wrapper Initialization
    seq_designer = LigandMPNNWrapper(python = "python", run_py = "LigandMPNN/run.py")

    return model, seq_designer

def design_sequence(designer,model_type,pdb_file,chains_to_design="A",omit_AA="C",bias_AA="",temperature=0.10,return_logits=False,
                    fixed_residues = "", seed = 111):
    """Runs the LigandMPNN (or SolubleMPNN) sequence design wrapper."""
    if seed is None:
        seed = int(np.random.randint(0, high = 99999, size = 1, dtype = int)[0])
    
    seq, logits = designer.run(
        model_type=model_type,
        pdb_path=pdb_file,
        seed=seed, # Fixed seed for reproducibility per design tool
        chains_to_design=chains_to_design,
        bias_AA=bias_AA,
        omit_AA=omit_AA,
        return_logits=return_logits,
        extra_args={
            "--temperature": temperature,
            "--batch_size": 1,
            "--fixed_residues" : fixed_residues,
        },
    )
    if return_logits:
        return seq, logits

    return seq[0], logits

def binder_binds_contacts(metrics, target_chain, epitope_residues, paratope_residues: str = ""):
    """
    Returns True if at least 2 contact residues on target_chain are contacted by the binder, using the
    paratope/epitope indices that model.analyze_structure() already computed (via analyze_structure_holo)
    for this target_chain — no separate structural analysis call needed here.
    """
    paratope_indices = [int(x[1:]) for x in paratope_residues.split(",") if x.strip()]
    epitope_indices = [int(x[1:]) for x in epitope_residues.split(",") if x.strip() and x[0] == target_chain]

    actual_paratope_indices = [int(x) for x in metrics[f'paratope_indices_{target_chain}'].split(",") if x.strip()]
    actual_epitope_indices = [int(x) for x in metrics[f'epitope_indices_{target_chain}'].split(",") if x.strip()]

    # Both paratope and epitope residues are optional constraints — if none apply to this call (either
    # not specified at all, or epitope_indices filtered down to empty because none of the specified
    # epitope_residues belong to this particular target_chain), don't gate on that side at all rather
    # than forcing a failure. Only require >=2 common residues on a side that actually has constraints.
    num_common_paratope_residues = len(set(paratope_indices).intersection(actual_paratope_indices))
    num_common_epitope_residues = len(set(epitope_indices).intersection(actual_epitope_indices))
    paratope_ok = (num_common_paratope_residues >= 2) if paratope_indices else True
    epitope_ok = (num_common_epitope_residues >= 2) if epitope_indices else True

    return paratope_ok and epitope_ok

def run_refine_cycle(model, seq_designer, args):
    """ 
    Cycle 0: Validate Predicted Structure of the inputs passes initial contact check
    Cycle 1 -> N: Sequence Design -> Structure Prediction
    Run this process for a number of cycles to refine the in-silico designed protein. The output of each cycle is saved in the output directory.

    """
    # Setup:
    target_chains = ",".join(chr(ord('B') + i) for i in range(len(model.seq_list) - 1))
    path_design_cycle_folder = os.path.join(args.path_output_dir, "design_cycles")
    path_improved_designs_folder = os.path.join(args.path_output_dir, "improved_insilico")
    if not os.path.exists(path_design_cycle_folder):
        os.makedirs(path_design_cycle_folder)
    if not os.path.exists(path_improved_designs_folder):
        os.makedirs(path_improved_designs_folder)

    # MPNN designs ligand-aware sequences if a ligand is part of the complex, otherwise plain soluble design
    model_type = "ligand_mpnn" if model.ligand_list else "soluble_mpnn"
    print("MPNN Model being used: ", model_type)

    # ---- Cycle 0: Predict structure for the initial (un-redesigned) sequence and validate it passes the contact check ----
    predicted_structure, _ = model.predict_structure()
    path_structure_cycle_0 = os.path.join(path_design_cycle_folder, f"{model.design_name}_cycle_0.cif")
    metrics_cycle_0 = model.analyze_structure(predicted_structure, model_id = 0, path_structure = path_structure_cycle_0)
    path_pdb_cycle_0 = convert_cif_to_pdb(path_structure_cycle_0)

    contact_check_res_cycle_0 = [
        binder_binds_contacts(metrics = metrics_cycle_0, target_chain = target_chain,
                               epitope_residues = args.epitope_residues, paratope_residues = args.paratope_residues)
        for target_chain in target_chains.split(",")
    ]
    contact_check_passed_cycle_0 = all(contact_check_res_cycle_0)
    iptm_cycle_0 = metrics_cycle_0["iptm"]
    print(f"Cycle 0: iptm={iptm_cycle_0:.4f}, contact_check_passed={contact_check_passed_cycle_0}")
    if not contact_check_passed_cycle_0:
        print("Cycle 0: initial design did not pass the contact check — continuing anyway, MPNN redesign may fix it.")

    # Best-cycle tracking starts from cycle 0's own analyzed iptm (if it passed the contact check),
    # rather than an arbitrary -inf placeholder.
    if contact_check_passed_cycle_0:
        best_iptm, best_cycle, best_seq, best_pdb_path = iptm_cycle_0, 0, model.seq_list[0], path_pdb_cycle_0
    else:
        best_iptm, best_cycle, best_seq, best_pdb_path = float("-inf"), None, None, None
    prev_pdb_path = path_pdb_cycle_0

    # ---- Cycles 1 -> N: Sequence Design -> Structure Prediction ----
    for cycle in range(1, args.num_cycles + 1):
        # 1. Design a new binder sequence via MPNN, conditioned on the previous cycle's structure
        seq_str, _ = design_sequence(seq_designer, model_type, pdb_file = prev_pdb_path, fixed_residues= args.fixed_residues,
                                            chains_to_design = "A", temperature = args.mpnn_temperature)
        print("MPNN_Derived_Binder_Seq: ", seq_str)
        new_binder_seq = seq_str.split(":")[0] 

        model.seq_list[0] = new_binder_seq

        # 2. Re-predict structure with the redesigned binder sequence, target(s) held fixed
        predicted_structure, yaml_input = model.predict_structure()

        # 3. Analyze: saves the CIF at our chosen per-cycle path, plus PAE, ptm/iptm/plddt, and (for
        # holo) contact/ipSAE metrics for every target chain in one call.
        path_structure_cycle = os.path.join(path_design_cycle_folder, f"{model.design_name}_cycle_{cycle}.cif")
        metrics = model.analyze_structure(predicted_structure, model_id = cycle, path_structure = path_structure_cycle)
        path_pdb_cycle = convert_cif_to_pdb(path_structure_cycle)

        # 4. Contact check for this cycle's structure, using analyze_structure()'s own metrics
        contact_check_res = [
            binder_binds_contacts(metrics = metrics, target_chain = target_chain,
                                   epitope_residues = args.epitope_residues, paratope_residues = args.paratope_residues)
            for target_chain in target_chains.split(",")
        ]
        contact_check_passed = all(contact_check_res)

        # 5. Track best cycle: must pass the contact check and improve on iptm
        current_iptm = metrics["iptm"]
        print(f"Cycle {cycle}: iptm={current_iptm:.4f}, contact_check_passed={contact_check_passed}")
        if contact_check_passed and current_iptm > best_iptm:
            best_iptm = current_iptm
            best_cycle, best_seq, best_pdb_path = cycle, new_binder_seq, path_pdb_cycle
            # Save improved designs to improved_insilico folder along with spec on how to create it
            best_cycle_design_name = f"{model.design_name}_cycle_{cycle}"
            shutil.copy2(path_pdb_cycle, os.path.join(path_improved_designs_folder, f"{best_cycle_design_name}.pdb"))
            with open(os.path.join(path_improved_designs_folder, f"{best_cycle_design_name}.yml"), 'w') as spec_file:
                yaml.dump(yaml_input, spec_file)
        prev_pdb_path = path_pdb_cycle

    return {
        "best_cycle": best_cycle,
        "best_seq": best_seq,
        "best_iptm": best_iptm if best_cycle is not None else None,
        "best_pdb_path": best_pdb_path,
    }


def main():
    parser = argparse.ArgumentParser(description = "Refine in-silico designed proteins via iterative structre prediction -> seq generation cycles.")

    # Required arguments
    parser.add_argument("seq_binder", type = str, 
                        help = "Binder sequence to refine")
    parser.add_argument("seq_target", type = str, 
                        help = "Target sequence or sequences to use in refinement. If multiple targets, separate with commas (e.g. 'target1,target2').")
    parser.add_argument("model_name", type = str, choices = ['ESMFold2', 'Boltz2', 'ESMFold2-Fast'], 
                        help = "Model to use for refinement.")
    parser.add_argument("design_name", type = str, 
                        help = "Name of the design to refine. This will be used to name the output files.")
    parser.add_argument("path_output_dir", type = str, 
                        help = "Path to the output directory where the refined designs will be saved.")

    # Optional arguments
    parser.add_argument("--num_cycles", type = int, default = 5)
    parser.add_argument("--ligands", type = str, default = "",
                        help = "Comma-separated string of ligands")
    parser.add_argument("--filename_output", type = str, default = "refined_designs.csv")
    parser.add_argument("--paratope_residues", type = str, default = "",
                        help = "Comma-separated string of residues on the binder that should interact with the target. e.g. 'A10,A11,A12'.")
    parser.add_argument("--epitope_residues", type = str, default = "",
                        help = "Comma-separated string of residues on the target that should interact with the binder. e.g. 'B10,B20,C30'.")
    parser.add_argument("--fixed_residues", type = str, default = "",
                        help = "Space-separated string of residues on the binder that should be fixed during MPNN seq redesign. e.g. A10 A11 A12 A13" )
    parser.add_argument("--mpnn_temperature", type = float, default = 0.1,
                        help = "Temperature to sample residues during seq redesign. Higher temperature -> greater volatility in the generated sequence")
    args = parser.parse_args()

    model, seq_designer = load_model_setup_run(model_name = args.model_name, design_name = args.design_name, seq_binder = args.seq_binder,
                                 seq_target = args.seq_target, path_output_dir = args.path_output_dir, ligands = args.ligands)

    result = run_refine_cycle(model = model, seq_designer = seq_designer, args = args)
    print("Refinement result:", result)

    df_result = pd.DataFrame([result])
    path_result_csv = os.path.join(args.path_output_dir, args.filename_output)
    df_result.to_csv(path_result_csv, index = False)
    print(f"Saved refinement summary to {path_result_csv}")

    return result


if __name__ == "__main__":
    main()