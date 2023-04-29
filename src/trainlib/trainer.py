import os
import logging
import pprint
import tqdm

import torch
import torch.nn as nn

from sklearn.metrics import roc_auc_score, average_precision_score
from torch.nn.parallel.data_parallel import DataParallel
from torch.utils.data import DataLoader

# from utils.trainer_utils import (
#     rename_logger,
#     should_stop_early
# )
# from datasets import (
#   Dataset,
#   TokenizedDataset,
#   MLMTokenizedDataset
# )


class Trainer:
    def __init__(self, args):
        self._args = args
        pass

