from abc import ABC, abstractclassmethod
import torch
import os
from typing import Any
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


class Callback(ABC):

    def __init__(self, **kwargs):
        pass

    @abstractclassmethod
    def reset_state(self) -> None:
        pass

    @abstractclassmethod
    def __call__(self, **kwargs):
        pass


class EarlyStopping():

    def __init__(self,
                 monitor='loss',
                 min_delta=0.0001,
                 patience=10,
                 mode='min'):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        self.best_epoch = 0

    def reset_state(self):
        self.counter = 0
        self.best_value = None
        self.best_epoch = 0
        self.early_stop = False

    def __call__(self, value, epoch) -> bool:

        # Initialise best_value value if none exists
        if self.best_value == None:
            self.best_value = value
            self.best_epoch = epoch + 1

        if self.mode == 'min':
            if self.best_value - value > self.min_delta:
                self.best_value = value
                self.best_epoch = epoch + 1
                self.counter = 0
            else:
                self.counter += 1
        elif self.mode == 'max':
            if value - self.best_value > self.min_delta:
                self.best_value = value
                self.best_epoch = epoch + 1
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            # Executes early stopping
            logger.info(
                f'Early stopping executed at epoch: {epoch+1} | best epoch: {self.best_epoch} | best_value: {self.best_value}'
            )
            return False

        return True


class ModelCheckpoint():

    def __init__(self,
                 monitor='loss',
                 save_best_only=False,
                 min_delta=0.0001,
                 mode='min'):
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = None
        self.best_epoch = 0
        self.best_metrics = None
        self.best_loss = 0.0

    def reset_state(self):
        self.best_epoch = 0
        self.best_value = None
        self.best_metrics = None
        self.best_loss = 0.0

    def _save_ckpt(self, ckpt_path, model, optimizer):

        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(
            {
                'epoch': self.best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)

    def __call__(self,
                 value,
                 epoch,
                 ckpt_path,
                 model,
                 optimizer,
                 metrics,
                 loss,
                 rank=0):

        # Initialise best_value value if none exists
        if self.best_value == None:
            self.best_value = value
            self.best_epoch = epoch + 1
            self.best_metrics = metrics
            self.best_loss = loss

            # Save model on the first epoch
            if rank == 0:
                self._save_ckpt(ckpt_path, model, optimizer)

        if not self.save_best_only:
            # If self.save_best_only is False, then save model on all epochs
            self.best_value = value
            self.best_epoch = epoch + 1
            self.best_metrics = metrics
            self.best_loss = loss

            if rank == 0:
                self._save_ckpt(ckpt_path, model, optimizer)
            return

        # If self.save_best_only is True, then check if model is the best
        if self.mode == 'min':
            # Save model if value lower than current best
            if self.best_value - value > self.min_delta:
                self.best_value = value
                self.best_epoch = epoch + 1
                self.best_metrics = metrics
                self.best_loss = loss

                if rank == 0:
                    self._save_ckpt(ckpt_path, model, optimizer)

        elif self.mode == 'max':
            # Save model if value higher than current best
            if value - self.best_value > self.min_delta:
                self.best_value = value
                self.best_epoch = epoch + 1
                self.best_metrics = metrics
                self.best_loss = loss

                if rank == 0:
                    self._save_ckpt(ckpt_path, model, optimizers)