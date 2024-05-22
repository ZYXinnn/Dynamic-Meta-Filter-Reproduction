# add ***********************************************************************
from __future__ import print_function
from __future__ import division
# end_add ***********************************************************************
# -*- coding: utf-8 -*-
import sys
sys.dont_write_bytecode = True

import torch
import os
from core.config import Config
from core import Trainer
# add ***********************************************************************
import time
import datetime
import argparse
import os.path as osp
import numpy as np
import random

import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.nn.functional as F
# sys.path.append('./torchFewShot')

# from args_xent import argument_parser
#
# from torchFewShot.models.net import Model
# from torchFewShot.data_manager import DataManager
# from torchFewShot.losses import CrossEntropyLoss
# from torchFewShot.optimizers import init_optimizer
#
# from torchFewShot.utils.iotools import save_checkpoint, check_isfile
# from torchFewShot.utils.avgmeter import AverageMeter
# from torchFewShot.utils.logger import Logger
# from torchFewShot.utils.torchtools import one_hot, adjust_learning_rate
# from torchFewShot.utils.lr_helper import warmup_scheduler

from tqdm import tqdm

# parser = argument_parser()
# args = parser.parse_args()
# end_add ***********************************************************************

def main(rank, config):
    trainer = Trainer(rank, config)
    trainer.train_loop(rank)


if __name__ == "__main__":
    config = Config("./config/proto.yaml").get_config_dict()

    if config["n_gpu"] > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["device_ids"]
        torch.multiprocessing.spawn(main, nprocs=config["n_gpu"], args=(config,))
    else:
        main(0, config)