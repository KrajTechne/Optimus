import argparse
import os
import shutil
import yaml
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless (no display) in a Modal container — must be set before pyplot import
import matplotlib.pyplot as plt

from LigandMPNN.wrapper import LigandMPNNWrapper
from StrucTools import convert_cif_to_pdb

_PLOT_METRICS = {'plddt': {'ymin' : 0.0, "ymax": 1.0, 'color' : "#FF7F11"},
                 'ptm': {'ymin' : 0.0,  'ymax': 1.0, 'color' : "#E94560"},
                 'iptm': {'ymin' : 0.0, 'ymax': 1.0, 'color' : "#9B59B6" },
                 'ipsae_min' : {'ymin' : 0.0, 'ymax' : 1.0, 'color' : "#2ECC71"},
}

# --threshold's actual default depends on --filter_metric (argparse can't express a default
# conditioned on another argument) — main() and Modal's _run_refiner_body both leave --threshold
# unset (None) and call resolve_threshold() to fill it in from here consistently.
_DEFAULT_THRESHOLDS = {"iptm": 0.8, "ipsae_min": 0.61}

def resolve_threshold(filter_metric: str, threshold) -> float:
    return _DEFAULT_THRESHOLDS[filter_metric] if threshold is None else threshold

def load_model_setup_run(
        model_name: str, 
        design_name: str, 
        seq_binder: str, 
        seq_target: str, 
        path_output_dir: str,
        ligands: str = "",
        epitope_residues: str = "",
        msa_options: str = "",
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
    if msa_options == "":
        msa_options = []
    else:
        msa_options = msa_options.split(',')

    # Structure Prediction Model Initialization
    # Imported lazily, per branch, rather than at module level: RunOpenDDE pins torch==2.7.1/cu126
    # while RunESMFold2/RunBoltz2's stack pins torch==2.11.0/cu130 — directly conflicting builds,
    # so no single environment can have all three frameworks installed at once. Importing only the
    # one actually needed keeps each model's own Modal image from requiring the others' deps.
    if model_name in ['ESMFold2', 'ESMFold2-Fast']:
        from RunESMFold2 import RunESMFold2
        model = RunESMFold2(design_name = design_name, model_name = model_name, seq_list = seq_list,
                            path_output_dir = path_output_dir, ligand_list= ligand_list,
                            desired_epitope_residues = desired_epitope_residues, msa_options= msa_options, **kwargs)
    elif model_name == 'Boltz2':
        from RunBoltz2 import RunBoltz2
        model = RunBoltz2(design_name = design_name, seq_list = seq_list, path_output_dir = path_output_dir,
                          ligand_list= ligand_list, desired_epitope_residues= desired_epitope_residues, msa_options= msa_options, **kwargs)
    elif model_name == 'OpenDDE':
        # msa_options drives RunOpenDDE's per-chain 'empty'/''/'.a3m path' handling same as
        # Boltz2/ESMFold2 — RunOpenDDE._resolve_use_msa() derives OpenDDE's own job-level
        # use_msa switch from it internally.
        from RunOpenDDE import RunOpenDDE
        model = RunOpenDDE(design_name = design_name, seq_list = seq_list, path_output_dir = path_output_dir,
                          ligand_list= ligand_list, desired_epitope_residues= desired_epitope_residues, msa_options= msa_options, **kwargs)
    else:
        raise ValueError(f"Model name {model_name} is not supported. Please choose from ['ESMFold2', 'Boltz2', 'ESMFold2-Fast', 'OpenDDE']")

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


def _normalize_plddt_key(metrics: dict) -> dict:
    """
    RunESMFold2 (holo)/RunBoltz2 key their mean-pLDDT-across-the-structure value as
    "complex_plddt"; RunOpenDDE keys the same quantity (confirmed from its own source,
    opendde/model/sample_confidence.py: `summary_confidence["plddt"] = atom_plddt.mean(dim=-1)
    * 100`) as "plddt" directly. Same value, different key — not two different metrics, so this
    normalizes onto "plddt" (the more semantically correct name, since pLDDT at the structure
    level is already a mean by definition, not a distinct "mean_plddt" quantity) rather than
    branching on model_name wherever this gets consumed later (e.g. the cycle-history plotting).
    """
    # Re-assign plddt to complex_plddt if provided as an option
    if "plddt" not in metrics and "complex_plddt" in metrics:
        metrics["plddt"] = metrics["complex_plddt"]
    # Normalize plddt numeric range
    if metrics['plddt'] > 1:
        metrics['plddt'] = metrics['plddt'] / 100
    return metrics


def _best_structure(predicted_structure):
    """
    RunESMFold2.predict_structure() returns a list of structures (one per diffusion sample)
    when num_samples > 1, pre-ranked by ESMFold2 itself — so the first entry is always its best
    sample, and analyze_structure() (which expects a single structure, not a list) should be
    called on that one. RunOpenDDE/RunBoltz2 always return None here regardless of num_samples
    (they read results back from disk instead — see their own predict_structure() docstrings),
    so this is a no-op for them.
    """
    return predicted_structure[0] if isinstance(predicted_structure, list) else predicted_structure


def run_refine_cycle(model, seq_designer, args, design_count):
    """ 
    Cycle 0: Validate Predicted Structure of the inputs passes initial contact check
    Cycle 1 -> N: Sequence Design -> Structure Prediction
    Run this process for a number of cycles to refine the in-silico designed protein. The output of each cycle is saved in the output directory.

    """
    # Setup: Create overarching design_cycles folder and improved_insilico folder
    target_chains = ",".join(chr(ord('B') + i) for i in range(len(model.seq_list) - 1))
    path_overarching_design_cycle_folder = os.path.join(args.path_output_dir, "runs")
    path_improved_designs_folder = os.path.join(args.path_output_dir, "improved_insilico")
    if not os.path.exists(path_overarching_design_cycle_folder):
        os.makedirs(path_overarching_design_cycle_folder)
    if not os.path.exists(path_improved_designs_folder):
        os.makedirs(path_improved_designs_folder)
    # Setup: Create design counter specific folder
    path_design_specific_folder = os.path.join(path_overarching_design_cycle_folder, f"run_{design_count}")
    if not os.path.exists(path_design_specific_folder):
        os.makedirs(path_design_specific_folder)

    # MPNN designs ligand-aware sequences if a ligand is part of the complex, otherwise plain soluble design
    model_type = "ligand_mpnn" if model.ligand_list else "soluble_mpnn"
    print("MPNN Model being used: ", model_type)

    # ---- Cycle 0: Predict structure for the initial (un-redesigned) sequence and validate it passes the contact check ----
    predicted_structure, _ = model.predict_structure()
    predicted_structure = _best_structure(predicted_structure)
    path_structure_cycle_0 = os.path.join(path_design_specific_folder, f"{model.design_name}_cycle_0.cif")
    metrics_cycle_0 = _normalize_plddt_key(model.analyze_structure(predicted_structure, path_structure = path_structure_cycle_0))
    path_pdb_cycle_0 = convert_cif_to_pdb(path_structure_cycle_0)

    contact_check_res_cycle_0 = [
        binder_binds_contacts(metrics = metrics_cycle_0, target_chain = target_chain,
                               epitope_residues = args.epitope_residues, paratope_residues = args.paratope_residues)
        for target_chain in target_chains.split(",")
    ]
    contact_check_passed_cycle_0 = all(contact_check_res_cycle_0)
    print(f"Cycle 0: {args.filter_metric}={metrics_cycle_0[args.filter_metric]:.4f}, contact_check_passed={contact_check_passed_cycle_0}")
    if not contact_check_passed_cycle_0:
        print("Cycle 0: initial design did not pass the contact check — continuing anyway, MPNN redesign may fix it.")

    prev_pdb_path = path_pdb_cycle_0

    # Tall (one row per cycle) history of every metric analyze_structure() computes, for plotting
    # confidence metrics over cycles later. run_id set directly from design_count (already a param
    # here) rather than looping over cycle_history afterward in the caller just to stamp it on.
    cycle_history = [{"run_id": design_count, "cycle": 0, "contact_check_passed": contact_check_passed_cycle_0, **metrics_cycle_0}]

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
        predicted_structure = _best_structure(predicted_structure)

        # 3. Analyze the structure
        path_structure_cycle = os.path.join(path_design_specific_folder, f"{model.design_name}_cycle_{cycle}.cif")
        metrics = _normalize_plddt_key(model.analyze_structure(predicted_structure, path_structure = path_structure_cycle))
        path_pdb_cycle = convert_cif_to_pdb(path_structure_cycle)

        # 4. Contact check for this cycle's structure, using analyze_structure()'s own metrics
        contact_check_res = [
            binder_binds_contacts(metrics = metrics, target_chain = target_chain,
                                   epitope_residues = args.epitope_residues, paratope_residues = args.paratope_residues)
            for target_chain in target_chains.split(",")
        ]
        contact_check_passed = all(contact_check_res)
        cycle_history.append({"run_id": design_count, "cycle": cycle, "contact_check_passed": contact_check_passed, **metrics})
        print(f"Cycle {cycle}: {args.filter_metric}={metrics[args.filter_metric]:.4f}, contact_check_passed={contact_check_passed}")

        # 5. Keep this cycle's design if it passes the contact check and clears args.threshold on args.filter_metric
        if contact_check_passed and metrics[args.filter_metric] >= args.threshold:
            design_name_cycle = f"{model.design_name}_run_{design_count}_cycle_{cycle}"
            shutil.copy2(path_pdb_cycle, os.path.join(path_improved_designs_folder, f"{design_name_cycle}.pdb"))
            with open(os.path.join(path_improved_designs_folder, f"{design_name_cycle}.yml"), 'w') as spec_file:
                yaml.dump(yaml_input, spec_file)
        prev_pdb_path = path_pdb_cycle

    plot_cycle_metrics_png(cycle_history, run_id=design_count, path_run_folder=path_design_specific_folder)
    return cycle_history


def plot_cycle_metrics_png(cycle_history: list[dict], run_id: int, path_run_folder: str) -> str:
    df_run = pd.DataFrame(cycle_history).sort_values("cycle")
    available_metrics = [m for m in _PLOT_METRICS if m.lower() in df_run.columns]
    if not available_metrics:
        print("plot_cycle_metrics_png: none of the expected metrics are present, skipping plot.")
        return ""

    fig, axes = plt.subplots(
        1, len(available_metrics), figsize=(3.5 * len(available_metrics), 3.5))

    for ax, metric in zip(axes, available_metrics):
        ax.set_facecolor("#fcfcfb")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#c3c2b7")
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
        ax.tick_params(colors="#898781", labelsize=8)

        # Plot points
        ax.plot(
            df_run["cycle"], df_run[metric],
            marker="o", markersize=5, linewidth=2, color=_PLOT_METRICS[metric]['color'], zorder=3,
        )
        # Add text of the raw value of the point
        for x, y in zip(df_run["cycle"], df_run[metric]):
            ax.annotate(
                f"{y:.2f}", xy=(x, y), xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=7, color="#52514e", clip_on=False,
            )

        ax.set_title(f"{metric} (Run {run_id})", color="#0b0b0b", fontsize=10)
        ax.set_xlabel("Cycle", color="#0b0b0b", fontsize=10)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_ylim(_PLOT_METRICS[metric]['ymin'], _PLOT_METRICS[metric]['ymax'])

    fig.tight_layout()

    path_png = os.path.join(path_run_folder, "cycle_metrics.png")
    fig.savefig(path_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved cycle metrics plot to {path_png}")
    return path_png


def iterate_over_design_count(args) -> pd.DataFrame:
    """ Run the cycling process for N design attempts and each one has K cycles"""
    all_cycle_records = []  # tall (run_id, cycle, <every analyze_structure() metric>) history, all design attempts
    for design_count in range(args.num_designs):
        # search_msa_every_cycle is an OpenDDE-only field (RunESMFold2/RunBoltz2 don't have it,
        # and would reject an unexpected kwarg) — only forwarded when actually running OpenDDE.
        extra_kwargs = {}
        if args.model_name == 'OpenDDE':
            extra_kwargs['search_msa_every_cycle'] = args.search_msa_every_cycle

        model, seq_designer = load_model_setup_run(model_name = args.model_name, design_name = args.design_name, seq_binder = args.seq_binder,
                                         seq_target = args.seq_target, path_output_dir = args.path_output_dir, ligands = args.ligands,
                                         epitope_residues= args.epitope_residues, msa_options = args.msa_options, num_samples = args.num_samples,
                                         **extra_kwargs)

        cycle_history = run_refine_cycle(model = model, seq_designer = seq_designer, args = args, design_count= design_count)
        all_cycle_records.extend(cycle_history)

    df_all_runs = pd.DataFrame(all_cycle_records)
    path_all_runs_csv = os.path.join(args.path_output_dir, "all_runs.csv")
    df_all_runs.to_csv(path_all_runs_csv, index = False)
    print(f"Saved per-cycle metric history to {path_all_runs_csv}")

    # Passing designs: every cycle (across every design attempt) that passed the contact check and
    # cleared args.threshold on args.filter_metric — not just a single "best" cycle per attempt.
    df_designs = df_all_runs[df_all_runs["contact_check_passed"] & (df_all_runs[args.filter_metric] >= args.threshold)]
    path_design_csv = os.path.join(args.path_output_dir, args.filename_output)
    df_designs.to_csv(path_design_csv, index = False)
    print(f"Saved {len(df_designs)} passing design(s) ({args.filter_metric} >= {args.threshold}) to {path_design_csv}")
    return path_design_csv

def main():
    parser = argparse.ArgumentParser(description = "Refine in-silico designed proteins via iterative structre prediction -> seq generation cycles.")

    # Required arguments
    parser.add_argument("seq_binder", type = str, 
                        help = "Binder sequence to refine")
    parser.add_argument("seq_target", type = str, 
                        help = "Target sequence or sequences to use in refinement. If multiple targets, separate with commas (e.g. 'target1,target2').")
    parser.add_argument("model_name", type = str, choices = ['ESMFold2', 'Boltz2', 'ESMFold2-Fast', 'OpenDDE'],
                        help = "Model to use for refinement.")
    parser.add_argument("design_name", type = str, 
                        help = "Name of the design to refine. This will be used to name the output files.")
    parser.add_argument("path_output_dir", type = str, 
                        help = "Path to the output directory where the refined designs will be saved.")

    # Optional arguments
    parser.add_argument("--num_cycles", type = int, default = 5,
                        help = "Number of cycles of seq-design -> structure prediction you want to do per design attempt")
    parser.add_argument("--num_designs", type = int, default = 1,
                        help = "Number of designs that you want to generate from initial binder sequence")
    parser.add_argument("--num_samples", type = int, default = 1,
                        help = "Number of structure-prediction samples per cycle (best-ranked one is used). Higher can improve accuracy at little/no extra runtime for some models (e.g. OpenDDE) since samples are batched on the GPU — worth checking per model before assuming it's free.")
    parser.add_argument("--search_msa_every_cycle", action = argparse.BooleanOptionalAction, default = True,
                        help = "OpenDDE only. True (default): real paired+unpaired MSA search every cycle via the public ColabFold API — correct but exposed to that server's occasional multi-minute PENDING queueing. False: cheaper cached/unpaired-only path (each unique sequence searched once, no pairing). Use --no-search_msa_every_cycle to disable.")
    parser.add_argument("--msa_options", type = str, default = "",
                        help = "MSA Options for structure prediction. Expecting comma-separated values of 'empty' or ''. The default runs with everything as 'empty'. "
                               "For OpenDDE specifically: recommended default is target-only search, e.g. 'empty,' for one binder+target — mark the binder 'empty' and only "
                               "the target(s) ''. Confirmed via replicate experiment that including the binder in the paired search scores lower on both iptm and actual "
                               "motif-ligand contact (see opendde_msa_findings_2026-09-14.md section 12).")
    parser.add_argument("--ligands", type = str, default = "",
                        help = "Comma-separated string of ligands")
    parser.add_argument("--filename_output", type = str, default = "top_designs.csv")
    parser.add_argument("--paratope_residues", type = str, default = "",
                        help = "Comma-separated string of residues on the binder that should interact with the target. e.g. 'A10,A11,A12'.")
    parser.add_argument("--epitope_residues", type = str, default = "",
                        help = "Comma-separated string of residues on the target that should interact with the binder. e.g. 'B10,B20,C30'.")
    parser.add_argument("--fixed_residues", type = str, default = "",
                        help = "Space-separated string of residues on the binder that should be fixed during MPNN seq redesign. e.g. A10 A11 A12 A13" )
    parser.add_argument("--mpnn_temperature", type = float, default = 0.1,
                        help = "Temperature to sample residues during seq redesign. Higher temperature -> greater volatility in the generated sequence")
    parser.add_argument("--filter_metric", type = str, choices = ['iptm', 'ipsae_min'], default = 'iptm',
                        help = "Structure-confidence metric used to decide which cycles' designs count as passing.")
    parser.add_argument("--threshold", type = float, default = None,
                        help = "Minimum --filter_metric value (plus passing the contact check) for a cycle to be kept as a passing "
                               "design. Defaults to 0.8 for iptm, 0.61 for ipsae_min if not set.")
    args = parser.parse_args()
    args.threshold = resolve_threshold(args.filter_metric, args.threshold)

    path_design_csv = iterate_over_design_count(args = args)

    return path_design_csv


if __name__ == "__main__":
    main()