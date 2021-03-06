from typing import Dict

import torch
from torch import nn

from model import model_components


class ModularModel(nn.Module):
    """Model with dynamic modules defined by the user during runtime 

    Attributes:
        regressor (Dict):
            Dict containing the config for the regressor module
        smiles_encoder (Dict):
            Dict containing the config for the module to encode the SMILES
            Default None
        sequence_encoder (Dict):
            Dict containing the config for the module to encode protein sequence
            Default None
        smiles (str):
            String representing the column label for the smiles
        sequence (str):
            String representing the column label for the sequence
    """

    def __init__(self, regressor_config: Dict, smiles_config: Dict,
                 sequence_config: Dict, smiles: str, sequence: str):
        super().__init__()

        # Initialise the smiles and sequence names
        self.smiles = smiles
        self.sequence = sequence

        # Initialise the smiles encoder
        self.smiles_encoder = getattr(
            model_components, smiles_config.pop('name'))(**smiles_config)

        # Initialise the sequence encoder
        self.seq_encoder = getattr(
            model_components, sequence_config.pop('name'))(**sequence_config)

        # Initialise the regressor
        self.reg = getattr(model_components,
                           regressor_config.pop('name'))(**regressor_config)

    def forward(self, inputs):
        encodes = []

        # Obtain the outputs from the smiles and sequence encoders
        encodes.append(self.smiles_encoder(inputs[self.smiles]))
        encodes.append(self.seq_encoder(inputs[self.sequence]))

        # Concats the encoder outputs and feed it into the regressor
        x = torch.cat(encodes, dim=-1)
        return torch.squeeze(self.reg(x))
