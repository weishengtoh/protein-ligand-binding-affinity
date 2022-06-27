import argparse
import logging
import random
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torchmetrics
import yaml
from prettytable import PrettyTable
from torch import nn, optim

from metrics.metrics import EnrichmentFactor, KendallTau
from model.basemodel import Models
from model.custom_model import CustomModel
from utils.callbacks import Callback, EarlyStopping, ModelCheckpoint
from utils.history import History
import wandb
import horovod.torch as hvd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Define the name of the columns
SEQUENCE = 'Target Sequence'
SMILES = 'SMILES'
LABEL = 'Label'


def run(args):

    # Determine number of threads to run on
    torch.set_num_threads(4)

    hvd.init()
    torch.cuda.set_device(hvd.local_rank())

    ### --- LOAD THE CONFIG FILE --- ###
    with open(args.config_path) as file:
        config = yaml.safe_load(file)

    if hvd.rank() == 0:
        run = wandb.init(project='Protein_Ligand_Binding_Affinity',
                         group=f'{config["architecture"]}_horovod',
                         job_type='prepro_train',
                         config=config)

    train_config = config['train']

    ### --- SET SEED FOR REPRODUCIBILITY -- ##
    set_seed(train_config['seed'])

    ### --- SELECT MODEL ARCHITECTURE --- ###
    architecture = Models(config['architecture'])

    ### --- CREATE MODEL --- ###
    model = CustomModel(architecture=architecture,
                        model_path=train_config['model_path'],
                        config=config)

    ### --- LOAD DATA --- ###
    df = pd.read_parquet(train_config['data_path'])
    dataloader_samplers = model.load_data(dataframe=df,
                                          smiles=SMILES,
                                          sequence=SEQUENCE,
                                          num_replicas=hvd.size(),
                                          rank=hvd.rank(),
                                          world_size=hvd.size(),
                                          label=LABEL,
                                          mode='train')
    train_dl, val_dl, test_dl = dataloader_samplers['dataloaders']
    train_sampler, val_sampler, test_sampler = dataloader_samplers['samplers']

    ### --- GET LOSS, OPTIM, METRICS --- ###
    loss = nn.MSELoss()
    optimizer = optim.Adam(model.get_parameters(),
                           lr=train_config['learning_rate'] * hvd.size())
    hvd.broadcast_parameters(model.model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.get_named_parameters())

    train_metrics = obtain_train_metrics()
    val_metrics = obtain_train_metrics()

    ### --- USE CALLBACKS (OPTIONAL) --- ###
    callbacks = obtain_callbacks()

    ### --- COMPILE MODEL --- ###
    model.compile(optimizer, loss, train_metrics, val_metrics)

    if hvd.rank() == 0:
        ### --- USE TENSORBOARD FOR LOGGING (OPTIONAL)--- ###
        model.init_tensorboard()

        ### --- USE WANDB FOR LOGGING --- ###
        model.init_wandb(train_dl)

    ### --- TRAIN AND VALIDATE --- ###
    train_history = model.train_dist(max_epochs=train_config['epochs'],
                                     train_data=train_dl,
                                     val_data=val_dl,
                                     local_rank=hvd.local_rank(),
                                     rank=hvd.rank(),
                                     train_sampler=train_sampler,
                                     val_sampler=val_sampler,
                                     callbacks=callbacks)
    print_results(train_history, hvd.rank())

    if hvd.rank() == 0 and test_dl is not None:
        ### --- LOG FINAL METRICS TO WANDB --- ###
        for key, value in train_history.get_params().items():
            wandb.summary[key] = value

    ### --- TEST (OPTIONAL)--- ###
    if hvd.rank() == 0 and test_dl is not None and len(test_dl) > 1:
        test_metrics = obtain_test_metrics()
        test_loss = nn.MSELoss()
        model.load_ckpt(model.model_path)
        test_history = model.evaluate(test_data=test_dl,
                                      loss_fn=test_loss,
                                      metrics=test_metrics)
        print_results(test_history, hvd.rank())

    # Save model to wandb as artifact
    if hvd.rank() == 0:
        ckpt_path = model.model_path
        artifact = wandb.Artifact(config['architecture'], type='model')
        artifact.add_file(ckpt_path)
        run.log_artifact(artifact)

        # Finish the wandb run
        wandb.finish()


def print_results(history: History, rank: int = 0, precision: int = 4) -> None:

    ### --- GENERATE TABLE TO STORE METRICS --- ###
    metrics_table = PrettyTable(['Metric', 'Value'])
    for key, value in history.get_params().items():
        metrics_table.add_row([key, f'{value:.{precision}f}'])

    ### --- PRINT OUT METRICS IN TERMINAL --- ###
    if rank == 0:
        print(metrics_table)


def obtain_train_metrics() -> Dict[str, Any]:

    metrics = {
        'rmse': torchmetrics.MeanSquaredError(squared=False),
        'pearson': torchmetrics.PearsonCorrCoef(),
        'r2score': torchmetrics.R2Score(),
        'mae': torchmetrics.MeanAbsoluteError()
    }

    return metrics


def obtain_test_metrics() -> Dict[str, Any]:

    metrics = {
        'rmse': torchmetrics.MeanSquaredError(squared=False),
        'pearson': torchmetrics.PearsonCorrCoef(),
        'r2score': torchmetrics.R2Score(),
        'mae': torchmetrics.MeanAbsoluteError(),
        'ef_1': EnrichmentFactor(1),
        'ef_5': EnrichmentFactor(5),
        'ef_10': EnrichmentFactor(10),
        'k_tau': KendallTau()
    }
    return metrics


def obtain_callbacks() -> List[Callback]:
    """Obtains the callbacks to be used during training"""

    earlystopping = EarlyStopping(monitor='loss',
                                  min_delta=0.0001,
                                  patience=20,
                                  mode='min')
    modelcheckpoint = ModelCheckpoint(monitor='loss',
                                      save_best_only=True,
                                      min_delta=0.0001,
                                      mode='min')

    return [earlystopping, modelcheckpoint]


def set_seed(seed):
    """Set the seed for reproducibility"""

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        '-C',
                        type=str,
                        help='Path to the config file',
                        required=True)
    args = parser.parse_args()
    run(args)
