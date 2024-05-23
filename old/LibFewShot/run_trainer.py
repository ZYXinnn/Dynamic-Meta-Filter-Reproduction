# add ***********************************************************************
# from __future__ import print_function
# from __future__ import division
# end_add ***********************************************************************
# -*- coding: utf-8 -*-
import sys
sys.dont_write_bytecode = True

import torch
import os
from core.config import Config
from core import Trainer

def main(rank, config):
    trainer = Trainer(rank, config)
    trainer.train_loop(rank)


if __name__ == "__main__":
    config = Config("./config/DynamicWeightsModel.yaml").get_config_dict()

    if config["n_gpu"] > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["device_ids"]
        torch.multiprocessing.spawn(main, nprocs=config["n_gpu"], args=(config,))
    else:
        main(0, config)