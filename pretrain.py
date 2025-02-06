#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/16 11:20
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : pretrain.py
# @Description :
import argparse
import sys
import wandb
import numpy as np
import torch
import torch.nn as nn
import copy
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import models, train
from config import MaskConfig, TrainConfig, PretrainModelConfig
from models import LIMUBertModel4Pretrain
from utils import set_seeds, get_device, get_sample_weights \
    , LIBERTDataset4Pretrain, handle_argv, load_pretrain_data_config, load_pretrain_config, prepare_classifier_dataset, \
    prepare_pretrain_dataset, prepare_datasets_participants, balance_dataset, Preprocess4Normalization,  Preprocess4Mask


def main(args, training_rate, balance=False, balance_ratio=0):

    wandb.init(project='pretraining', entity='spgaryf')


    if args.dataset != 'c24':
        data, labels, train_cfg, model_cfg, mask_cfg, dataset_cfg = load_pretrain_data_config(args)
    else:
        train_cfg, model_cfg, mask_cfg, dataset_cfg = load_pretrain_config(args)

    wandb.config.balance = balance
    wandb.config.balance_ratio = balance_ratio
    wandb.config.hidden = model_cfg.hidden
    wandb.config.n_layers = model_cfg.n_layers

    if args.dataset != 'c24':
        pipeline = [Preprocess4Normalization(model_cfg.feature_num), Preprocess4Mask(mask_cfg)]
    else: # C24 is already in Gs
        pipeline = [Preprocess4Mask(mask_cfg)]
    # pipeline = [Preprocess4Mask(mask_cfg)]
    
    if args.dataset != 'c24':
        data_train, label_train, data_vali, _ = prepare_pretrain_dataset(data, labels, training_rate, seed=train_cfg.seed)
    else:
        data_train, label_train, data_vali, _, _, _ = prepare_datasets_participants(args, training_rate, seed=train_cfg.seed)
        if balance:
            data_train, label_train = balance_dataset(data_train, label_train, balance_ratio)

    print("data train shape is", data_train.shape)
    print("data vali shape is", data_vali.shape)
    print("label train shape is", label_train.shape)
    if data_vali.shape[0] > data_train.shape[0]:
        print("shuffling and cutting")
        np.random.shuffle(data_vali)
        num_samples = int(0.1*data_train.shape[0])
        data_vali = data_vali[:num_samples]
        print("new data vali shape is", data_vali.shape)



    ## Sampler dataloader
    unique_ytrain, counts_ytrain = np.unique(label_train, return_counts=True)
    print('y_train label distribution: ', dict(zip(unique_ytrain, counts_ytrain)))
    weights = 100.0 / torch.Tensor(counts_ytrain)
    weights = weights.double()
    print('weights of sampler: ', weights)
    sample_weights = get_sample_weights(label_train, weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    data_set_train = LIBERTDataset4Pretrain(data_train, pipeline=pipeline)
    data_loader_train = DataLoader(data_set_train, shuffle=False, batch_size=train_cfg.batch_size, sampler=sampler)


    data_set_vali = LIBERTDataset4Pretrain(data_vali, pipeline=pipeline)
    data_loader_vali = DataLoader(data_set_vali, shuffle=False, batch_size=train_cfg.batch_size)
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
        trainer.pretrain(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_vali,
                         model_file=args.pretrain_model)
    else:
        trainer.pretrain(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_vali, model_file=None)

if __name__ == "__main__":
    mode = "base"
    balance = False
    balance_ratio = 100
    args = handle_argv('pretrain_' + mode, 'pretrain.json', mode)
    training_rate = 0.8
    main(args, training_rate, balance = balance, balance_ratio = balance_ratio)
