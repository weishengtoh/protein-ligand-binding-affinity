import argparse
import logging
import random
from pathlib import Path
from typing import Any, Dict, Union
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch import nn
import torchmetrics
import yaml
import os
from prettytable import PrettyTable
from utils.history import History
from metrics.metrics import EnrichmentFactor, KendallTau
from model.custom_model import CustomModel, Models


def run(args):

    ### --- LOAD THE CONFIG FILE --- ###
    with open(args.config_path) as file:
        config = yaml.safe_load(file)

    predict_config = config['predict']

    ### --- SET SEED FOR REPRODUCIBILITY -- ##
    set_seed(predict_config['seed'])

    ### --- SELECT MODEL ARCHITECTURE --- ###
    architecture = Models(config['architecture'])

    ### --- GET MODEL --- ###
    model = CustomModel(architecture=architecture, config=config)

    ### --- LOAD DATA --- ###
    df = pd.read_parquet(predict_config['data_path'])
    dataloader_samplers = model.load_data(dataframe=df,
                                          smiles=predict_config['smiles'],
                                          sequence=predict_config['sequence'],
                                          label=predict_config['label'],
                                          mode='predict')
    dataloader, _, _ = dataloader_samplers['dataloaders']

    ### --- LOAD MODEL FROM CK_PT -- ##
    model.load_ckpt(predict_config['ck_pt'])

    ### --- GENERATE PREDICTIONS --- ###
    preds = model.predict(dataloader)

    # ### --- SAVE PREDICTIONS --- ###
    df['Pred'] = preds
    save_predictions(df)

    ### --- GENERATE METRICS (OPTIONAL) --- ###
    if len(dataloader) > 1:
        test_metrics = obtain_metrics()
        loss = nn.MSELoss()
        test_history = model.evaluate(dataloader, loss, test_metrics)
        print_results(test_history)


def save_predictions(dataframe: pd.DataFrame,
                     path: Union[str, Path] = 'outputs') -> None:
    """Saves the results from dataframe to csv"""

    os.makedirs(path, exist_ok=True)
    str_date_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    dataframe.to_csv(os.path.join('outputs', f'{str_date_time}.csv'))


def print_results(history: History, precision: int = 4) -> None:

    ### --- GENERATE TABLE TO STORE METRICS --- ###
    metrics_table = PrettyTable(['Metric', 'Value'])
    for key, value in history.get_params().items():
        metrics_table.add_row([key, f'{value:.{precision}f}'])

    ### --- PRINT OUT METRICS IN TERMINAL --- ###
    print(metrics_table)


def obtain_metrics() -> Dict[str, Any]:

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
