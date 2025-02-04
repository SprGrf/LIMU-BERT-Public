#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/16 11:20
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : pretrain.py
# @Description :
import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import copy
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import models, train
from config import MaskConfig, TrainConfig, PretrainModelConfig
from models import LIMUBertModel4Pretrain
from utils import set_seeds, get_device \
    , LIBERTDataset4Pretrain, handle_argv, load_pretrain_data_config, load_pretrain_config, prepare_classifier_dataset, \
    prepare_pretrain_dataset, prepare_datasets_participants, balance_dataset, Preprocess4Normalization,  Preprocess4Mask


def main(args, training_rate):
    original = False

    if original:
        data, labels, train_cfg, model_cfg, mask_cfg, dataset_cfg = load_pretrain_data_config(args)
    else:
        train_cfg, model_cfg, mask_cfg, dataset_cfg = load_pretrain_config(args)

    pipeline = [Preprocess4Normalization(model_cfg.feature_num), Preprocess4Mask(mask_cfg)]
    # pipeline = [Preprocess4Mask(mask_cfg)]
    
    if original:
        data_train, label_train, data_test, label_test = prepare_pretrain_dataset(data, labels, training_rate, seed=train_cfg.seed)
    else:
        data_train, label_train, data_test, label_test, _, _ = prepare_datasets_participants(args, training_rate, seed=train_cfg.seed)
        balanced = True
        if balanced:
            data_train, label_train = balance_dataset(data_train, label_train, 1)

    data_set_train = LIBERTDataset4Pretrain(data_train, pipeline=pipeline)
    data_set_test = LIBERTDataset4Pretrain(data_test, pipeline=pipeline)
    data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size)
    data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size)
    model = LIMUBertModel4Pretrain(model_cfg)

    criterion = nn.MSELoss(reduction='none')

    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=train_cfg.lr, momentum=0.9)


    device = get_device(args.gpu)
    trainer = train.Trainer(train_cfg, model, optimizer, args.save_path, device)

    def func_loss(model, batch):
        mask_seqs, masked_pos, seqs = batch #
        seq_recon = model(mask_seqs, masked_pos) #
        loss_lm = criterion(seq_recon, seqs) # for masked LM
        return loss_lm

    def func_forward(model, batch):
        mask_seqs, masked_pos, seqs = batch
        seq_recon = model(mask_seqs, masked_pos)
        return seq_recon, seqs

    def func_evaluate(seqs, predict_seqs):
        loss_lm = criterion(predict_seqs, seqs)
        return loss_lm.mean().cpu().numpy()

    if hasattr(args, 'pretrain_model'):
        print("Starting pretraining...")
        trainer.pretrain(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test,
                         model_file=args.pretrain_model)
    else:
        trainer.pretrain(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test, model_file=None)


if __name__ == "__main__":
    mode = "base"
    args = handle_argv('pretrain_' + mode, 'pretrain.json', mode)
    training_rate = 0.8
    main(args, training_rate)
