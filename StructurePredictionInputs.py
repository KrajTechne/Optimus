"""
StructurePredictionInputs — a worked example of a Pydantic dataclass.

Why a Pydantic *dataclass* instead of a `BaseModel`?
- You want the familiar `@dataclass` shape (plain attributes, auto __init__,
  auto __repr__) but with Pydantic's validation/coercion layered on top.
- It stays interoperable with the stdlib `dataclasses` module — things like
  `dataclasses.fields(...)` and `dataclasses.asdict(...)` still work.
- Past __init__, it behaves like any normal Python class: you add methods,
  properties, and classmethods exactly as you would on a plain dataclass.
  Pydantic only intercepts construction to validate/coerce inputs.

Run directly to see validation + methods in action:
    python StructurePredictionInputs.py
"""
import os
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic.dataclasses import dataclass


class ChainType(str, Enum):
    PROTEIN = "protein"
    DNA = "dna"
    RNA = "rna"
    LIGAND = "ligand"


@dataclass
class ChainInput:
    """One entity to fold — e.g. a binder chain, a target chain, or a ligand."""

    chain_id: str
    sequence: str
    chain_type: ChainType = ChainType.PROTEIN

    # A field_validator runs whenever `sequence` is set during construction.
    # Raising ValueError anywhere in here becomes a pydantic.ValidationError.
    @field_validator("sequence")
    @classmethod
    def sequence_must_be_letters(cls, v: str) -> str:
        cleaned = v.strip().upper()
        if not cleaned.isalpha():
            raise ValueError(f"sequence must contain only letters, got: {v!r}")
        return cleaned  # returned value becomes the stored field value (normalization)

    # Ordinary property — nothing Pydantic-specific here.
    @property
    def length(self) -> int:
        return len(self.sequence)


@dataclass
class StructurePredictionInputs:
    """ Traditional set of inputs for any protein structure prediction model """

    design_name: str = Field(default = "") # Name of the design being evaluated
    seq_list: list[ChainInput] = Field(default_factory=list) # List of sequence inputs for the design
    template_list: list[ChainInput] = Field(default_factory=list) # List of template paths for the design
    msa_options: list[str] = Field(default_factory=list) # List of MSA options for the design (either 'empty', '', or a .a3m file path)
    entity_types: list[ChainType] = Field(default_factory=list) # List of entity types for the design: Either protein, dna, or rna
    ligand_list: list[ChainInput] = Field(default_factory=list) # List of ligand inputs for the design
    num_samples: int = Field(default=1) # Number of model samples to generate for the design
    seed: int = Field(default=0) # Random seed for reproducibility of the model samples
    path_output_dir: str = Field(default = "") # Directory where the output files will be saved

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

    @field_validator("entity_types")
    @classmethod
    def validate_entity_types(cls, entity_types: list[ChainType]) -> list[ChainType]:
        """ Validate that the entity_types list contains only valid ChainType values"""
        for entity_type in entity_types:
            if entity_type not in ChainType:
                raise ValueError(f"entity_types must be one of {list(ChainType)}, got: {entity_type!r}")
        return entity_types

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


    






    