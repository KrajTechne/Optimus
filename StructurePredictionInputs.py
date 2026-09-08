"""
StructurePredictionInputs: 
    Collects & Validates inputs for structure prediction models, including sequences, templates, MSA options, entity types, ligands, and output directory
    Provides methods to analyze predicted structures of complexes for binding interfaces and ipSAE metrics.
"""
from __future__ import annotations

import os

from typing import Literal
from pydantic import Field, field_validator, model_validator
from pydantic.dataclasses import dataclass
from StrucTools import determine_binding_interface, calculate_ipSAE


@dataclass
class StructurePredictionInputs:
    """ Traditional set of inputs for any protein structure prediction model """

    design_name: str = Field(default = "") # Name of the design being evaluated
    seq_list: list[str] = Field(default_factory=list) # List of sequences for the design
    template_list: list[str] = Field(default_factory=list) # List of template paths for the design
    msa_options: list[str] = Field(default_factory=list) # List of MSA options for the design (either 'empty', '', or a .a3m file path)
    entity_types: list[Literal["protein", "dna", "rna"]] = Field(default_factory=list) # List of entity types for the design: Either protein, dna, or rna
    ligand_list: list[str] = Field(default_factory=list) # List of ligand SMILES strings for the design
    num_samples: int = Field(default=1) # Number of model samples to generate for the design
    seed: int = Field(default=0) # Random seed for reproducibility of the model samples
    path_output_dir: str = Field(default = "") # Directory where the output files will be saved
    desired_epitope_residues: list[str] = Field(default_factory=list) # List of desired epitope residues for the design (e.g. ["A10", "B20", "C30"])

    @field_validator("design_name")
    @classmethod
    def create_design_name(cls, v: str) -> str:
        """ If no design name is provided, create a default one based on the current timestamp and save in the output directory"""
        if not v:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            v = f"design_{timestamp}"
        return v

    @field_validator("msa_options")
    @classmethod
    def validate_msa_options(cls, v: list[str]) -> list[str]:
        """ Validate that the msa_options list contains only valid options: either 'empty', '', or a .a3m file path"""
        for option in v:
            if option not in ["empty", ""] and not option.endswith(".a3m"):
                raise ValueError(f"msa_options must be either 'empty', '', or a .a3m file path, got: {option!r}")
        return v

    @model_validator(mode = "after")
    def validate_output_dir(self) -> StructurePredictionInputs:
        """ Validate that the output_dir is a valid directory path. 
            If the path is empty, create a new directory in the current working directory with the design name.
        """
        if self.path_output_dir == "":
            path_parent_output_dir = os.getcwd()
            self.path_output_dir = os.path.join(path_parent_output_dir, f"structure_{self.design_name}")
        if not os.path.exists(self.path_output_dir):
            os.makedirs(self.path_output_dir)
        return self

    def analyze_structure_holo(self, path_structure: str, path_pae: str) -> dict:
        """ Analyze holo structure by doing contact check and calculating ipSAE
            Return a dictionary with the results of the analysis 
        """
        metrics = {}
        num_targets = len(self.seq_list) - 1

        # 1. Conduct contact check
        target_chains = ','.join(chr(ord('B') + i) for i in range(num_targets))
        print("target_chains: ", target_chains)
        for target_chain_id in target_chains.split(','):
            hotspots = [hotspot[1:] for hotspot in self.desired_epitope_residues if hotspot[0] == target_chain_id]
            print("Target chain: ", target_chain_id)
            contact_information = determine_binding_interface(pdb_file_path= path_structure,
                                                              hotspots= hotspots,
                                                              binder_chain_id= "A", target_chain_id= target_chain_id)

            # Append binding interface contacts information
            metrics.update(contact_information)

        # 2. Calculate ipSAE min and DockQ for each target chain
        ipsae_dict = calculate_ipSAE(pae_file = path_pae,
                                     binder_chain = "A",
                                     target_chains = target_chains,
                                     path_input_structure = path_structure)
        ipsae_values = [value for key, value in ipsae_dict.items() if key.startswith("ipSAE_")]
        if ipsae_values:
            ipsae_dict["ipsae_min"] = min(ipsae_values) # Min of the ipsae_min for all target chains in the complex
            ipsae_dict["ipsae_max"] = max(ipsae_values) # Max of the ipsae_min for all target chains in the complex
        metrics.update(ipsae_dict)

        return metrics



    
    