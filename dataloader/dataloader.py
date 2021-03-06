import logging

import torch
from typing import Union, Optional, Tuple, Callable
from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader as geo_dataloader
from torch.utils.data import DataLoader as py_dataloader
from torch.utils.data import Dataset, random_split
from torch.utils.data.distributed import DistributedSampler

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


class DataLoaderGen:

    @staticmethod
    def load_data(data_path: Union[str, Path]) -> pd.DataFrame:
        """Loads the data from parquet file to pandas DataFrame"""
        dataframe = pd.read_parquet(data_path)
        return dataframe

    @staticmethod
    def preprocess_data(dataframe: pd.DataFrame,
                        smiles: str,
                        sequence: str,
                        label: str = None) -> pd.DataFrame:
        """Simple preprocessing function to remove all invalid/missing data"""

        dataframe = dataframe.copy()

        # Drop all columns not used
        drop_columns = [
            col for col in dataframe.columns
            if (col not in [smiles, sequence, label])
        ]

        dataframe.drop(columns=drop_columns, inplace=True)

        # Find the indices where there are inf or nan in label
        if label is not None:
            indices = []

            for idx, val in enumerate(dataframe[label]):
                if np.isinf(val).any() or np.isnan(val).any():
                    indices.append(idx)

            # Drop the rows with inf or nan values
            dataframe.drop(index=indices, inplace=True)

        dataframe.reset_index(drop=True, inplace=True)

        return dataframe

    @staticmethod
    def get_dataloaders(dataframe: pd.DataFrame,
                        prepro_fn: Callable,
                        smiles: str,
                        sequence: str,
                        num_replicas: int = None,
                        rank: int = None,
                        label: str = None,
                        world_size: int = 1,
                        val_ratio: Optional[float] = None,
                        test_ratio: Optional[float] = None,
                        drop_last: bool = True,
                        batch_size: int = 32,
                        shuffle: bool = True,
                        collate_fn: Callable = None):
        """Returns a Dictionary of dataloaders and samplers(optional)"""

        # Create the custom dataset
        dataset = CustomDataset(dataframe=dataframe,
                                smiles=smiles,
                                sequence=sequence,
                                label=label,
                                transform=prepro_fn)

        # Find the size of the datasets
        val_size = int(val_ratio * len(dataset))
        test_size = int(test_ratio * len(dataset))
        train_size = len(dataset) - val_size - test_size

        # DGL models must use the standard Pytorch dataloaders
        if collate_fn is not None:
            dataloader = py_dataloader
        else:
            dataloader = geo_dataloader

        # Samplers only required if running on Horovod
        train_sampler, val_sampler, test_sampler = None, None, None

        if val_size != 0 and test_size != 0:
            ### --- GENERATE TRAIN/VAL/TEST SPLITS --- ###

            train_ds, val_ds, test_ds = random_split(
                dataset, [train_size, val_size, test_size])

            if num_replicas is not None and rank is not None:
                train_sampler = DistributedSampler(train_ds,
                                                   num_replicas=num_replicas,
                                                   rank=rank)
                val_sampler = DistributedSampler(val_ds,
                                                 num_replicas=num_replicas,
                                                 rank=rank)

            train_dl = dataloader(
                train_ds,
                batch_size=batch_size,
                drop_last=drop_last,
                shuffle=shuffle if train_sampler is None else False,
                collate_fn=collate_fn,
                sampler=train_sampler)
            val_dl = dataloader(
                val_ds,
                batch_size=batch_size,
                drop_last=drop_last,
                shuffle=shuffle if val_sampler is None else False,
                collate_fn=collate_fn,
                sampler=val_sampler)
            test_dl = dataloader(test_ds,
                                 batch_size=batch_size,
                                 drop_last=drop_last,
                                 shuffle=shuffle,
                                 collate_fn=collate_fn)

        elif val_size != 0 and test_size == 0:
            ### --- GENERATE TRAIN/VAL SPLIT ONLY --- ###

            train_ds, val_ds = random_split(dataset, [train_size, val_size])

            if num_replicas is not None and rank is not None:
                train_sampler = DistributedSampler(train_ds,
                                                   num_replicas=num_replicas,
                                                   rank=rank)
                val_sampler = DistributedSampler(val_ds,
                                                 num_replicas=num_replicas,
                                                 rank=rank)

            train_dl = dataloader(
                train_ds,
                batch_size=batch_size,
                drop_last=drop_last,
                shuffle=shuffle if train_sampler is None else False,
                collate_fn=collate_fn,
                sampler=train_sampler)
            val_dl = dataloader(
                val_ds,
                batch_size=batch_size,
                drop_last=drop_last,
                shuffle=shuffle if val_sampler is None else False,
                collate_fn=collate_fn,
                sampler=val_sampler)
            test_dl = None
        else:
            ### --- GENERATE TRAIN SPLIT ONLY --- ###

            if num_replicas is not None and rank is not None:
                train_sampler = DistributedSampler(train_ds,
                                                   num_replicas=num_replicas,
                                                   rank=rank)
            train_dl = dataloader(
                dataset,
                batch_size=batch_size,
                drop_last=drop_last,
                shuffle=shuffle if train_sampler is None else False,
                collate_fn=collate_fn,
                sampler=train_sampler)
            val_dl = None
            test_dl = None

        ### --- GENERATE LOGS DESCRIBING THE DATA SPLITS --- ###
        DataLoaderGen._get_logs(train_dl=train_dl,
                                batch_size=batch_size,
                                train_size=train_size,
                                drop_last=drop_last,
                                world_size=world_size,
                                val_dl=val_dl,
                                test_dl=test_dl,
                                val_size=val_size,
                                test_size=test_size)

        return {
            'dataloaders': (train_dl, val_dl, test_dl),
            'samplers': (train_sampler, val_sampler, test_sampler)
        }

    @staticmethod
    def _get_logs(train_dl: Union[geo_dataloader, py_dataloader],
                  batch_size: int,
                  train_size: int,
                  drop_last: bool,
                  world_size: int = 1,
                  val_dl: Union[geo_dataloader, py_dataloader] = None,
                  test_dl: Union[geo_dataloader, py_dataloader] = None,
                  val_size: int = None,
                  test_size: int = None):
        if val_dl is not None and test_dl is not None:

            logger.info(f'''
            Dataset Information (batch size: {batch_size} | world_size: {world_size} | drop last: {drop_last})
            Train | Instances: {train_size:,} | Batches/Process: {len(train_dl):,}
            Val   | Instances: {val_size:,} | Batches/Process: {len(val_dl):,}
            Test  | Instances: {test_size:,} | Batches/Process: {len(test_dl):,}
            ''')

        elif val_dl is not None:

            logger.info(f'''
            Dataset Information (batch size: {batch_size:,} | world_size: {world_size} | drop last: {drop_last})
            Train | Instances: {train_size:,} | Batches/Process: {len(train_dl):,}
            Val   | Instances: {val_size:,} | Batches/Process: {len(val_dl):,}
            ''')
        else:

            logger.info(f'''
            Dataset Information (batch size: {batch_size:,} | world_size: {world_size} | drop last: {drop_last})
            Instances: {train_size:,} | Batches/Process: {len(train_dl):,}
            ''')


class CustomDataset(Dataset):
    """Define a custom pytorch dataset"""

    def __init__(self,
                 dataframe: pd.DataFrame,
                 smiles: str,
                 sequence: str,
                 label: str = None,
                 transform: Callable = None):
        self.dataframe = dataframe.copy()
        self.transform = transform
        self.smiles = smiles
        self.sequence = sequence
        self.label = label

    def __len__(self) -> int:
        """Returns the length of the dataset"""
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> torch.Tensor:

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Obtain the row(s) of data at position idx
        sample = {}
        for key in [self.sequence, self.smiles]:

            feature = self.dataframe[key].iloc[idx]
            sample[key] = feature

        # Add the label to the sample dict
        if self.label is not None:
            label = self.dataframe[self.label].iloc[idx]
            sample[self.label] = np.array(label, dtype=np.float32)

        # Apply the transform if required
        if self.transform:
            sample = self.transform(sample)

        return sample