from torchmetrics import Metric
import torch
import numpy as np
from scipy.stats import kendalltau


class EnrichmentFactor(Metric):

    def __init__(self, percent: int):
        super().__init__()
        assert percent <= 100 and percent >= 0, 'Percent has to be within 0 - 100'
        self.percent = percent
        self.add_state('preds', default=[], dist_reduce_fx="sum")
        self.add_state('targets', default=[], dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):

        self.preds.append(preds)
        self.targets.append(targets)

    def compute(self) -> torch.Tensor:

        preds = np.concatenate(tuple(x.cpu().numpy() for x in self.preds))
        targets = np.concatenate(tuple(x.cpu().numpy() for x in self.targets))

        pred_rank = np.argsort(preds).tolist()[::-1]
        true_rank = np.argsort(targets).tolist()[::-1]

        sample_percentage = self.percent / 100
        n_sample = int(sample_percentage * len(targets))
        if n_sample == 0:
            return torch.tensor(np.nan)

        conc_sample = len([
            i for i in pred_rank[:n_sample] if i in true_rank[:n_sample]
        ]) / n_sample

        return torch.tensor(conc_sample / sample_percentage,
                            dtype=torch.float64)


class KendallTau(Metric):

    def __init__(self, alternative: str = 'two-sided'):
        super().__init__()
        assert alternative in [
            'two-sided', 'less', 'greater'
        ], 'KendallTau alternative must be one of ["two-sided", "less", "greater"]'
        self.alternative = alternative
        self.add_state('preds', default=[], dist_reduce_fx="sum")
        self.add_state('targets', default=[], dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):

        self.preds.append(preds)
        self.targets.append(targets)

    def compute(self) -> torch.Tensor:

        preds = np.concatenate(tuple(x.cpu().numpy() for x in self.preds))
        targets = np.concatenate(tuple(x.cpu().numpy() for x in self.targets))

        corr, pvalue = kendalltau(preds, targets, alternative=self.alternative)

        return torch.tensor(corr, dtype=torch.float64)
