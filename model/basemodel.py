from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Union, Tuple
import pandas as pd

import numpy as np
import yaml
import os
import torch
from torch import nn, optim
from torch_geometric.loader import DataLoader
from torchmetrics import Metric
from utils.history import History
from enum import Enum, unique
from datetime import datetime


@unique
class Models(Enum):
    GCN_CNN = 'gcn_cnn'
    CNN_CNN = 'cnn_cnn'
    CNN_TRANSFORMERS = 'cnn_transformers'
    GAT_CNN = 'gat_cnn'
    GAT_TRANSFORMERS = 'gat_transformers'
    GCN_TRANSFORMERS = 'gcn_transformers'
    GCN_ESM1B = 'gcn_esm1b'
    GCN_PROTBERT = 'gcn_protbert'
    GINCONTEXTPRED_CNN = 'gincontextpred_cnn'
    GINEDGEPRED_CNN = 'ginedgepred_cnn'
    GININFOMAX_CNN = 'gininfomax_cnn'
    GINMASK_CNN = 'ginmask_cnn'
    GINCONTEXTPRED_PROTBERT = 'gincontextpred_protbert'
    GINEDGEPRED_PROTBERT = 'ginedgepred_protbert'
    GINMASK_PROTBERT = 'ginmask_protbert'


class BaseModel(ABC):
    """"Abstract base class for all the models"""

    def __init__(self,
                 architecture: Models,
                 config: Dict,
                 model_path: Union[str, Path] = None):

        self.architecture = architecture
        self.date_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        if model_path is not None:
            os.makedirs(model_path, exist_ok=True)
            self.model_path = os.path.join(
                *[model_path, self.architecture.value, self.date_time])

        # Loads the model configuration
        self.config = config

        # Select gpu if available
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    @abstractmethod
    def load_data(
            self,
            dataframe: pd.DataFrame,
            smiles: str,
            sequence: str,
            label: str = None,
            mode: str = 'train') -> Tuple[DataLoader, DataLoader, DataLoader]:
        pass

    @abstractmethod
    def get_parameters(self) -> torch.Tensor:
        pass

    @abstractmethod
    def _get_prepro(self) -> Callable:
        pass

    @abstractmethod
    def train(self,
              max_epochs: int,
              train_data: DataLoader,
              val_data: Optional[DataLoader] = None,
              initial_epoch: int = 0,
              callbacks=None) -> History:
        """Abstract method used to train and validate(optional) the model."""

        pass

    @abstractmethod
    def evaluate(self, data: DataLoader, loss: nn,
                 metrics: Dict[str, Metric]) -> History:
        """Abstract method used to evaluate the model against a test set"""
        pass

    @abstractmethod
    def load_ckpt(self, path: Union[str, Path]) -> None:
        pass

    @abstractmethod
    def predict(self, data: DataLoader) -> Iterable[np.float32]:
        """Abstract method used to generate predictions"""
        pass

    @abstractmethod
    def compile(self,
                optimizer: optim,
                loss: nn,
                train_metrics: Dict[str, Metric],
                val_metrics: Dict[str, Metric] = None) -> None:
        """Abstract method to compile the model for training"""
        pass
