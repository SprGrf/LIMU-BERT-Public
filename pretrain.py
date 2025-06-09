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
    , LIBERTDataset4Pretrain, handle_argv, load_pretrain_config, \
    prepare_datasets_participants, balance_dataset, Preprocess4Normalization,  Preprocess4Mask, \
    Preprocess4Rotation, Preprocess4Scaling, Preprocess4Negation, Preprocess4TimeWarp, Preprocess4Flip, Preprocess4Shuffle


def main(args, training_rate, balance=False, balance_ratio=0, loocv=False, round=None):

    train_cfg, model_cfg, mask_cfg, dataset_cfg = load_pretrain_config(args)

    norm_acc = False
    pipeline = [Preprocess4Normalization(model_cfg.feature_num, norm_acc=norm_acc),
            Preprocess4Rotation(), Preprocess4Mask(mask_cfg)]
        
    data_train, label_train, data_vali, _, _, _ = prepare_datasets_participants(args, training_rate, seed=train_cfg.seed, loocv=loocv, round=round)
    if balance:
        data_train, label_train = balance_dataset(data_train, label_train, balance_ratio)


    ## Sampler dataloader
    unique_ytrain, counts_ytrain = np.unique(label_train, return_counts=True)
    print('y_train label distribution: ', dict(zip(unique_ytrain, counts_ytrain)))
    data_set_train = LIBERTDataset4Pretrain(data_train, pipeline=pipeline)

    
    # ## Weighted sampler dataloader
    weights = 100.0 / torch.Tensor(counts_ytrain)
    weights = weights.double()
    print('weights of sampler: ', weights)
    sample_weights = get_sample_weights(label_train, weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    data_loader_train = DataLoader(data_set_train, shuffle=False, batch_size=train_cfg.batch_size, sampler=sampler)   


    ## Normal Dataloader
    # data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size)

    data_set_vali = LIBERTDataset4Pretrain(data_vali, pipeline=pipeline)
    data_loader_vali = DataLoader(data_set_vali, shuffle=False, batch_size=train_cfg.batch_size)
    model = LIMUBertModel4Pretrain(model_cfg)

    criterion = nn.MSELoss(reduction='none')

    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=train_cfg.lr, momentum=0.9)


    device = get_device(args.gpu)
    if round != None:
        save_path = args.save_path + "_round_" + str(round)
    else:
        save_path = args.save_path
    trainer = train.Trainer(train_cfg, model, optimizer, save_path, device)

    def func_loss(model, batch):
        mask_seqs, masked_pos, seqs, _ = batch
        seq_recon = model(mask_seqs, masked_pos)
        loss_lm = criterion(seq_recon, seqs)
        return loss_lm

    def weighted_func_loss(model, batch, batch_weights):
        mask_seqs, masked_pos, seqs, _ = batch
        seq_recon = model(mask_seqs, masked_pos)
        loss_lm = criterion(seq_recon, seqs)
        # print(loss_lm)
        loss_lm = loss_lm.mean(dim=1)
        # print(loss_lm)
        weighted_loss = loss_lm * batch_weights.unsqueeze(1)
        return weighted_loss

    def func_forward(model, batch):
        mask_seqs, masked_pos, seqs, _ = batch
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
        trainer.pretrain(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_vali, 
                         model_file=None)

if __name__ == "__main__":
    mode = "base"
    balance = False
    balance_ratio = 1
    args = handle_argv('pretrain_' + mode, 'pretrain.json', mode)
    training_rate = 0.8
    set_seeds(10)
    if args.case_study == "cv":
        loocv = False
        if args.dataset == "C24":
            rounds = 1 # just one round with the presplit C24, 100 for training, 51 for testing
            print("C24, only doing one round...")            
        else:
            rounds = 10
        if args.dataset_cfg.user_label_size <= 10 or (args.dataset_cfg.user_ids and len(args.dataset_cfg.user_ids) <= 10):                
            print("Applying L.O.O.CV.")
            loocv = True
        total_users = len(args.dataset_cfg.user_ids) if args.dataset_cfg.user_ids else args.dataset_cfg.user_label_size
        print("total users are ",total_users)
        for round in range(min(total_users,rounds)):
            print("ROUND ", round)              
            main(args, training_rate, balance = balance, balance_ratio = balance_ratio, loocv=loocv, round=round)
    elif args.case_study == "d2d":
        args.save_path += "_d2d"
        main(args, training_rate, balance = balance, balance_ratio = balance_ratio)
