#!/usr/bin/python

import os
import sys
import argparse
import random
import numpy as np
import os.path as osp

import torch
import torch.nn as nn

from model_loop import RefineT5, TransformerT5
from utils_loop import *
        

def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="A two-stage framework for predicting GEL")

    parser.add_argument("-d", dest="data_dir", type=str, default="/your/path/TSSF/Human-Data50/K562")
    parser.add_argument("-n", dest="name", type=str, default="K562")
    parser.add_argument("-c", dest="checkpoint", type=str, default='/your/path/TSSF/models_human_50/K562')
    parser.add_argument("-g", dest="gpu", type=str, default='0,1')
    parser.add_argument("-f", dest="fea_dim", type=int, default=4)

    return parser.parse_args()


def main():
    """Create the model and start the training."""
    args = get_args()
    data_dir = args.data_dir
    name = args.name
    gpu = args.gpu
    # during the warming-up stage
    loss_lowest = float('inf')
    manual_seed = 2222
    if torch.cuda.is_available():
        torch.cuda.manual_seed(manual_seed)
    else:
        torch.manual_seed(manual_seed)
    trials = 15
    state_dict = None
    for trial in range(trials):
        model = TransformerT5(fea_dim=args.fea_dim)
        # if existing multiple GPUs, and using DataParallel
        if len(gpu.split(',')) > 1 and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=[int(id_) for id_ in gpu.split(',')])
        warmup = Trainer(model=model, data_dir=data_dir, checkpoint=args.checkpoint)
        loss, state_dict_cur = warmup.train(warmup_state=True)
        if loss_lowest > loss:
            print("The current loss is {:.3f}. Store the model in the {}-th trial.\n".format(loss, trial+1))
            loss_lowest = loss
            state_dict = state_dict_cur
    # during the training stage
    f_out = open(osp.join(args.checkpoint, 'score.txt'), 'a')
    # loading the warming-up model
    model.load_state_dict(state_dict, strict=True)
    train = Trainer(model=model, data_dir=data_dir, checkpoint=args.checkpoint)
    loss, _ = train.train()
    f_out.write("{}\ttrain_loss:{:.3f}\n".format(name, loss))
    # during the test stage
    checkpoint_file = osp.join(args.checkpoint, 'model.best.pth')
    chk = torch.load(checkpoint_file, map_location='cuda:0')
    state_dict_te = chk['model_state_dict']
    model = RefineT5(fea_dim=args.fea_dim)
    # if existing multiple GPUs, and using DataParallel
    if len(gpu.split(',')) > 1 and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[int(id_) for id_ in gpu.split(',')])
    # loading the training model
    model.load_state_dict(state_dict_te, strict=True)
    test = Trainer(model=model, data_dir=data_dir, checkpoint=args.checkpoint)
    pr, mae = test.test()
    f_out.write("{}\tpr: {:.3f}\tmae: {:.3f}\n".format(name, pr, mae))
    f_out.close()


if __name__ == "__main__":
    main()

