#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/16 11:20
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : classifier.py
# @Description :
import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

import train
from config import  load_dataset_label_names
from embedding import load_embedding_label
from models import fetch_classifier
from plot import plot_matrix

from statistic import stat_acc_f1, stat_acc_f1, stat_acc_f1_rec, stat_results
from utils import get_device, handle_argv \
    , IMUDataset, load_classifier_config, prepare_classifier_dataset


def classify_embeddings(args, label_index, training_rate, label_rate, balance=False, method=None):
    train_cfg, model_cfg, dataset_cfg = load_classifier_config(args)
    print(model_cfg)
    label_names, label_num = load_dataset_label_names(dataset_cfg, label_index)


    # #######################################################3
    # data_train, label_train, data_vali, label_vali, data_test, label_test \
    #     = prepare_classifier_dataset(data, labels, label_index=label_index, training_rate=training_rate
    #                                  , label_rate=label_rate, merge=model_cfg.seq_len, seed=train_cfg.seed
    #                                  , balance=balance)
    # print("After preparation of classifier data shape")
    # print(data_train.shape)
    # print(data_vali.shape)
    # print(data_test.shape)
    # data_set_train = IMUDataset(data_train, label_train)
    # data_set_vali = IMUDataset(data_vali, label_vali)
    # data_set_test = IMUDataset(data_test, label_test)
    # data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size)
    # data_loader_vali = DataLoader(data_set_vali, shuffle=False, batch_size=train_cfg.batch_size)
    # data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size)
    
    
    # unique_label_train, counts_train = np.unique(label_train, return_counts=True)
    # unique_label_vali, counts_vali = np.unique(label_vali, return_counts=True)
    # unique_label_test, counts_test = np.unique(label_test, return_counts=True)
    # print('Train label distribution: ', dict(zip(unique_label_train, counts_train)))
    # print('Validation label distribution: ', dict(zip(unique_label_vali, counts_vali)))
    # print('Test label distribution: ', dict(zip(unique_label_test, counts_test)))
    # ##############################################################
    
    criterion = nn.CrossEntropyLoss()
    model = fetch_classifier(method, model_cfg, input=72, output=label_num)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)  # , weight_decay=0.95
    trainer = train.Trainer(train_cfg, model, optimizer, args.save_path, get_device(args.gpu))

    def func_loss(model, batch):
        inputs, label = batch
        logits = model(inputs, True)
        loss = criterion(logits, label)
        return loss

    def func_forward(model, batch):
        inputs, label = batch
        logits = model(inputs, False)
        return logits, label

    def func_evaluate(label, predicts):
        stat = stat_acc_f1_rec(label.cpu().numpy(), predicts.cpu().numpy())
        return stat

    # trainer.train(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test, data_loader_vali)
    label_test, label_estimate_test = trainer.train_parts(args, func_loss, func_forward, func_evaluate, label_rate, label_index, balance, model_file=args.pretrain_model)    
    # label_estimate_test = trainer.run(func_forward, None, data_loader_test)
    return label_test, label_estimate_test


if __name__ == "__main__":

    training_rate = 0.8 # unlabeled sample / total sample
    label_rate = 1.0 # labeled sample / unlabeled sample
    balance = False

    mode = "base"
    method = "gru"
    args = handle_argv('classifier_' + mode + "_" + method, 'train.json', method)
    label_test, label_estimate_test = classify_embeddings(args, args.label_index,
                                                          training_rate, label_rate, balance=balance, method=method)

    label_names, label_num = load_dataset_label_names(args.dataset_cfg, args.label_index)
    acc, matrix, f1 = stat_results(label_test, label_estimate_test)
    matrix_norm = plot_matrix(matrix, label_names)


