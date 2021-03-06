import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import horovod.torch as hvd
import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from dataloader.dataloader import DataLoaderGen
from preprocess.preprocess import PreprocessGen
from prettytable import PrettyTable
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Metric, MetricCollection
from tqdm.auto import tqdm
from utils.callbacks import Callback, EarlyStopping, ModelCheckpoint
from utils.history import History

from model.basemodel import BaseModel, Models
from model.modular_model import ModularModel

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


class CustomModel(BaseModel):

    def __init__(self,
                 architecture: Models,
                 config: Dict,
                 model_path: Union[str, Path] = None):
        super().__init__(architecture, config, model_path)
        self.use_wandb = False
        self.writer = None

    def load_data(self,
                  dataframe: pd.DataFrame,
                  smiles: str,
                  sequence: str,
                  num_replicas: int = None,
                  rank: int = None,
                  world_size: int = 1,
                  label: str = None,
                  mode: str = 'train'):

        self.smiles = smiles
        self.sequence = sequence
        self.label = label
        self.config['model'].update({
            'smiles': smiles,
            'sequence': sequence,
        })

        # Define the model in pytorch nn.Module
        self.model = ModularModel(**self.config['model'])

        df = DataLoaderGen.preprocess_data(dataframe, smiles, sequence, label)

        if mode == 'train':
            dataloader_samplers = DataLoaderGen.get_dataloaders(
                dataframe=df,
                prepro_fn=self._get_prepro(),
                smiles=smiles,
                sequence=sequence,
                num_replicas=num_replicas,
                rank=rank,
                label=label,
                world_size=world_size,
                val_ratio=self.config['train']['val_ratio'],
                test_ratio=self.config['train']['test_ratio'],
                drop_last=self.config['train']['drop_last'],
                batch_size=self.config['train']['batch_size'],
                shuffle=self.config['train']['shuffle'],
                collate_fn=self._get_collate_fn())
        elif mode == 'predict':
            dataloader_samplers = DataLoaderGen.get_dataloaders(
                dataframe=df,
                smiles=smiles,
                sequence=sequence,
                num_replicas=num_replicas,
                rank=rank,
                label=label,
                world_size=world_size,
                prepro_fn=self._get_prepro(),
                val_ratio=0,
                test_ratio=0,
                drop_last=False,
                batch_size=self.config['predict']['batch_size'],
                shuffle=False,
                collate_fn=self._get_collate_fn())

        return dataloader_samplers

    def get_parameters(self) -> torch.Tensor:
        return self.model.parameters()

    def get_named_parameters(self) -> torch.Tensor:
        return self.model.named_parameters()

    def _get_prepro(self) -> Callable:
        """Returns the preprocess function to be used with pytorch dataset."""

        return PreprocessGen.get_prepro_fn(smiles=self.smiles,
                                           sequence=self.sequence,
                                           label=self.label,
                                           model=self.architecture)

    def _get_collate_fn(self) -> Callable:

        return PreprocessGen.get_collate_fn(model=self.architecture)

    def compile(self,
                optimizer: optim,
                loss_fn: nn,
                train_metrics: Dict[str, Metric],
                val_metrics: Dict[str, Metric] = None) -> None:
        """Configures the optimizer, loss and metrics"""

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_metrics = MetricCollection(train_metrics)
        self.val_metrics = MetricCollection(val_metrics)

    def train(self,
              max_epochs: int,
              train_data: DataLoader,
              val_data: Optional[DataLoader] = None,
              initial_epoch: int = 0,
              callbacks: List[Callback] = None) -> History:
        """Performs training and evaluation"""

        self.training = True

        # Prints out the number of parameters in the model
        self._count_parameters(self.model)

        # Shift model, metrics to GPU if available
        self.model = self.model.to(self.device)
        self.train_metrics.to(self.device)
        if self.val_metrics is not None:
            self.val_metrics.to(self.device)

        ### --- START OF EPOCH --- ###
        for epoch in tqdm(range(initial_epoch, initial_epoch + max_epochs),
                          position=0,
                          desc="Epoch",
                          leave=True,
                          colour='green',
                          ncols=80):

            # Check whether to continue training
            if self.training == False:
                break

            # Init/reset the batch loss
            self.train_batch_loss = []
            self.val_batch_loss = []

            # Reset the train/val metrics
            self.train_metrics.reset()
            if self.val_metrics is not None:
                self.val_metrics.reset()

            ### --- START OF TRAINING --- ###
            for batch, data in enumerate(
                    tqdm(train_data,
                         position=1,
                         desc='Training Batch',
                         leave=False,
                         colour='red',
                         ncols=80)):
                self._train_batch(data)

            # Compute the train epoch level metrics and losses
            epoch_train_metrics = self.train_metrics.compute()
            epoch_train_loss = np.sum(self.train_batch_loss) / len(
                self.train_batch_loss)

            ### --- END OF TRAINING --- ###

            # Skip validation if validation data not provided
            if val_data is not None:
                ### --- START OF VALIDATION --- ###
                for batch, data in enumerate(
                        tqdm(val_data,
                             position=2,
                             desc='Validation Batch',
                             leave=False,
                             colour='blue',
                             ncols=80)):
                    self._evaluate_batch(data)

                # Compute the validation epoch level metrics and losses
                epoch_val_metrics = self.val_metrics.compute()
                epoch_val_loss = np.sum(self.val_batch_loss) / len(
                    self.val_batch_loss)

                ### --- END OF VALIDATION --- ###

            # Execute callbacks (if any)
            if callbacks is not None:
                epoch_metrics = epoch_val_metrics if val_data is not None else epoch_train_metrics
                epoch_loss = epoch_val_loss if val_data is not None else epoch_train_loss
                self._execute_callbacks(epoch, callbacks, epoch_metrics,
                                        epoch_loss)

            # Logs the metrics to TensorBoard / wandb
            self._generate_logs(epoch, epoch_train_metrics, epoch_train_loss,
                                epoch_val_loss, epoch_val_metrics)
            ### --- END OF EPOCH --- ###

        ### --- END OF TRAINING --- ##
        results = {
            'epoch': epoch,
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss
        }
        for name, value in epoch_val_metrics.items():
            results.update({f'val_{name}': value.cpu().numpy()})
        for name, value in epoch_train_metrics.items():
            results.update({f'train_{name}': value.cpu().numpy()})

        ### --- OVERWRITE FINAL RESULTS WITH BEST MODEL IF MODELCHECKPOINT EXISTS --- ###
        for cb in callbacks:
            if isinstance(cb, ModelCheckpoint):
                best_results = {
                    'best_epoch': cb.best_epoch,
                    'best_loss': cb.best_loss,
                }

                for name, value in cb.best_metrics.items():
                    best_results.update({f'best_{name}': value.cpu().numpy()})

                results.update(best_results)

        history = History(self.architecture.value, self.config)
        history.set_params(results)

        return history

    def evaluate(self, test_data: DataLoader, loss_fn: nn,
                 metrics: Dict[str, Metric]) -> History:
        """Perform evaluation on the dataset"""

        # Shift metrics and model to GPU if avail
        self.model.to(self.device)
        self.model.eval()  # Set model to eval mode

        # Initialise the loss and metrics
        self.loss_fn = loss_fn
        self.val_metrics = MetricCollection(metrics)
        self.val_metrics.to(self.device)
        self.val_batch_loss = []
        self.val_metrics.reset()

        # Run evaluation loop on batches
        for batch, data in enumerate(
                tqdm(test_data,
                     position=0,
                     desc="Batch",
                     leave=True,
                     colour='green',
                     ncols=80)):

            self._evaluate_batch(data)

        epoch_val_metrics = self.val_metrics.compute()
        epoch_val_loss = np.sum(self.val_batch_loss) / len(self.val_batch_loss)

        results = {'test_loss': epoch_val_loss}
        for name, value in epoch_val_metrics.items():
            results.update({f'test_{name}': value.cpu().numpy()})

        history = History(self.architecture.value, self.config)
        history.set_params(results)

        return history

    def load_ckpt(self, path: Union[str, Path], training=False) -> None:
        """Loads the model from checkpoint to continue training"""

        ckpt = torch.load(path, map_location=self.device)

        self.model.load_state_dict(ckpt['model_state_dict'])
        if training:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            for state in self.optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(self.device)

    def predict(self, test_data: DataLoader) -> Iterable[Any]:
        """Predicts on the dataset"""

        # Shift metrics and model to GPU if avail
        self.model.to(self.device)
        self.model.eval()  # Set model to eval mode

        preds = []
        # Run evaluation loop on batches
        for batch, data in enumerate(
                tqdm(test_data,
                     position=0,
                     desc="Batch",
                     leave=True,
                     colour='green',
                     ncols=80)):

            # Shift data to GPU if available
            data = {key: value.to(self.device) for key, value in data.items()}

            with torch.no_grad():
                output = self.model(data)  # Obtain the model outputs
                np_output = output.cpu().numpy()

                if np_output.size > 1:
                    preds.extend(np_output)
                else:
                    preds = np_output

        return preds

    def _train_batch(self, data: Dict[str, Any]) -> None:

        self.model.train()  # Set model to training mode

        # Shift data to GPU if available
        data = {key: value.to(self.device) for key, value in data.items()}

        # Compute batch outputs and loss
        output = self.model(data)
        loss = self.loss_fn(output, data[self.label])

        # Compute batch metrics and loss
        self.train_metrics.update(output, data[self.label])
        self.train_batch_loss.append(loss.item())

        # Backwards propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _evaluate_batch(self, data: Dict[str, Any]) -> None:

        self.model.eval()  # Set model to eval mode

        # Shift data to GPU if available
        data = {key: value.to(self.device) for key, value in data.items()}

        with torch.no_grad():

            # Compute batch outputs and loss
            output = self.model(data)
            loss = self.loss_fn(output, data[self.label])

            # Compute batch metrics and loss
            self.val_metrics.update(output, data[self.label])
            self.val_batch_loss.append(loss.item())

    def _generate_logs(self,
                       epoch: int,
                       epoch_train_metrics: MetricCollection,
                       epoch_train_loss: float,
                       epoch_val_loss: float = None,
                       epoch_val_metrics: MetricCollection = None) -> None:
        """Logs the training information with TensorBoard/Weights and Biases"""

        ### --- LOG THE TRAINING METRICS --- ###
        # Log the epoch level training metrics
        for name, value in epoch_train_metrics.items():
            if self.writer is not None:
                self.writer.add_scalars(name, {'train': value}, epoch + 1)
            if self.use_wandb:
                wandb.log({"epoch": epoch + 1, f'train_{name}': value})

        # Log the epoch level training loss
        if self.writer is not None:
            self.writer.add_scalars('epoch_loss', {'train': epoch_train_loss},
                                    epoch + 1)
        if self.use_wandb:
            wandb.log({"epoch": epoch + 1, f'train_loss': epoch_train_loss})

        ### --- SKIP LOGGING VAL METRICS IF NOT AVAILABLE --- ###
        if epoch_val_loss is None or epoch_val_metrics is None:
            return

        ### --- LOG THE VALIDATION METRICS --- ###
        # Log the epoch level validation metrics
        for name, value in epoch_val_metrics.items():
            if self.writer is not None:
                self.writer.add_scalars(name, {'val': value}, epoch + 1)
            if self.use_wandb:
                wandb.log({"epoch": epoch + 1, f'val_{name}': value})

        # Log the epoch level validation loss
        if self.writer is not None:
            self.writer.add_scalars('epoch_loss', {'val': epoch_val_loss},
                                    epoch + 1)
        if self.use_wandb:
            wandb.log({"epoch": epoch + 1, f'val_loss': epoch_val_loss})

    def _count_parameters(self, model):
        """Counts the number of parameters in a model"""

        table = PrettyTable(['Modules', 'Parameters'])
        table.align['Modules'] = 'l'
        table.align['Parameters'] = 'r'
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            table.add_row([name, f'{params:,}'])
            total_params += params
        print(table)
        print(f'Total Trainable Params: {total_params:,}')

    def _execute_callbacks(self,
                           epoch,
                           callbacks,
                           epoch_metrics,
                           epoch_loss,
                           rank=0):
        """Executes all the callbacks"""

        for cb in callbacks:
            ### --- EXECUTES EARLY STOPPING CALLBACK --- ###
            if isinstance(cb, EarlyStopping):
                value = epoch_metrics.get(cb.monitor) if epoch_metrics.get(
                    cb.monitor) is not None else epoch_loss
                self.training = cb(value, epoch)

            ### EXECUTES MODEL CHECKPOINT CALLBACK --- ###
            if isinstance(cb, ModelCheckpoint):
                value = epoch_metrics.get(cb.monitor) if epoch_metrics.get(
                    cb.monitor) is not None else epoch_loss
                cb(value,
                   epoch,
                   self.model_path,
                   self.model,
                   self.optimizer,
                   epoch_metrics,
                   epoch_loss,
                   rank=rank)

    def init_tensorboard(self) -> None:
        """Initialise TensorBoard writer"""

        os.makedirs('logs', exist_ok=True)
        path = os.path.join(*['logs', self.architecture.value, self.date_time])
        self.writer = SummaryWriter(path)

    def init_wandb(self, train_data) -> None:
        """Initialise Weights and Biases logging"""

        self.use_wandb = True

        batch_per_epoch = len(train_data)
        wandb.define_metric('epoch')
        wandb.define_metric('train_*', step_metric='epoch')
        wandb.define_metric('val_*', step_metric='epoch')
        wandb.watch(self.model,
                    self.loss_fn,
                    log="all",
                    log_freq=batch_per_epoch)

    def train_dist(self,
                   max_epochs: int,
                   train_data: DataLoader,
                   val_data: Optional[DataLoader] = None,
                   initial_epoch: int = 0,
                   local_rank: int = None,
                   rank: int = None,
                   train_sampler: DistributedSampler = None,
                   val_sampler: DistributedSampler = None,
                   callbacks: List[Callback] = None) -> History:
        """Performs training and evaluation"""

        self.device = torch.device(local_rank)

        self.training = True

        # Prints out the number of parameters in the model
        if rank == 0:
            self._count_parameters(self.model)

        # Shift model, metrics to GPU if available
        self.model = self.model.to(self.device)
        self.train_metrics.to(self.device)
        if self.val_metrics is not None:
            self.val_metrics.to(self.device)

        ### --- START OF EPOCH --- ###
        for epoch in tqdm(range(initial_epoch, initial_epoch + max_epochs),
                          position=0,
                          desc=f"Rank: {rank} | Epoch: ",
                          leave=True,
                          colour='green',
                          ncols=80):

            # Check whether to continue training
            if rank == 0:
                training = self.training
            else:
                training = True

            # If rank 0 terminates, all must terminate
            self.training = hvd.broadcast_object(training, root_rank=0)

            if self.training == False:
                break

            train_sampler.set_epoch(epoch)

            # Init/reset the batch loss
            self.train_batch_loss = []
            self.val_batch_loss = []

            # Reset the train/val metrics
            self.train_metrics.reset()
            if self.val_metrics is not None:
                self.val_metrics.reset()

            ### --- START OF TRAINING --- ###
            for batch, data in enumerate(
                    tqdm(train_data,
                         position=1,
                         desc=f'Rank: {rank} | Training Batch: ',
                         leave=False,
                         colour='red',
                         ncols=80)):
                self._train_batch(data)

            # Compute the train epoch level metrics and losses
            epoch_train_metrics = self.train_metrics.compute()
            epoch_train_loss = np.sum(self.train_batch_loss) / len(
                self.train_batch_loss)

            # Collate the metrics and loss from Horovod
            for key, value in epoch_train_metrics.items():
                avg_tensor = hvd.allreduce(value)
                epoch_train_metrics[key] = avg_tensor

            epoch_train_loss = torch.tensor(epoch_train_loss)
            epoch_train_loss = hvd.allreduce(epoch_train_loss)

            ### --- END OF TRAINING --- ###

            # Skip validation if validation data not provided
            if val_data is not None:

                val_sampler.set_epoch(epoch)
                ### --- START OF VALIDATION --- ###
                for batch, data in enumerate(
                        tqdm(val_data,
                             position=2,
                             desc=f'Rank: {rank} | Validation Batch: ',
                             leave=False,
                             colour='blue',
                             ncols=80)):
                    self._evaluate_batch(data)

                # Compute the validation epoch level metrics and losses
                epoch_val_metrics = self.val_metrics.compute()
                epoch_val_loss = np.sum(self.val_batch_loss) / len(
                    self.val_batch_loss)

                # Collate the metrics and loss from Horovod
                for key, value in epoch_val_metrics.items():
                    avg_tensor = hvd.allreduce(value)
                    epoch_val_metrics[key] = avg_tensor

                epoch_val_loss = torch.tensor(epoch_val_loss)
                epoch_val_loss = hvd.allreduce(epoch_val_loss)

                ### --- END OF VALIDATION --- ###

            # Execute callbacks (if any)
            if callbacks is not None:
                epoch_metrics = epoch_val_metrics if val_data is not None else epoch_train_metrics
                epoch_loss = epoch_val_loss if val_data is not None else epoch_train_loss

                self._execute_callbacks(epoch,
                                        callbacks,
                                        epoch_metrics,
                                        epoch_loss,
                                        rank=rank)

            # Logs the metrics to TensorBoard / wandb
            self._generate_logs(epoch, epoch_train_metrics, epoch_train_loss,
                                epoch_val_loss, epoch_val_metrics)
            ### --- END OF EPOCH --- ###

        ### --- END OF TRAINING --- ##
        results = {
            'last_epoch': epoch,
            'last_train_loss': epoch_train_loss,
            'last_val_loss': epoch_val_loss
        }
        for name, value in epoch_val_metrics.items():
            results.update({f'last_val_{name}': value.cpu().numpy()})
        for name, value in epoch_train_metrics.items():
            results.update({f'last_train_{name}': value.cpu().numpy()})

        ### --- OVERWRITE FINAL RESULTS WITH BEST MODEL IF MODELCHECKPOINT EXISTS --- ###
        for cb in callbacks:
            if isinstance(cb, ModelCheckpoint):
                best_results = {
                    'best_epoch': cb.best_epoch,
                    'best_loss': cb.best_loss,
                }

                for name, value in cb.best_metrics.items():
                    best_results.update({f'best_{name}': value.cpu().numpy()})

                results.update(best_results)

        history = History(self.architecture.value, self.config)
        history.set_params(results)

        return history
