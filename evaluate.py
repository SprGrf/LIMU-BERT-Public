#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/16 11:20
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : classifier_bert.py
# @Description :
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
import wandb
import train
from config import load_dataset_label_names
from models import BERTClassifier, fetch_classifier
from plot import plot_matrix

from statistic import stat_acc_f1_rec, stat_results
from utils import get_device,  handle_argv, get_sample_weights, separate_data_and_labels_by_user, \
    IMUDataset, load_bert_classifier_data_config, load_bert_classifier_config, Preprocess4Normalization, \
    prepare_classifier_dataset, prepare_datasets_participants, balance_dataset


def evaluate(args, label_index, training_rate, label_rate, frozen_bert=False, balance=True, balance_ratio=100):

    train_cfg, model_bert_cfg, model_classifier_cfg, dataset_cfg = load_bert_classifier_config(args)
    
    label_names, label_num = load_dataset_label_names(dataset_cfg, label_index)

    _, _, _, _, data_test, label_test_full = prepare_datasets_participants(args, training_rate, seed=train_cfg.seed)

    label_test = label_test_full[:, 0, args.dataset_cfg.activity_label_index]    
 
    norm_acc = False if args.dataset == 'C24' else True
    pipeline = [Preprocess4Normalization(model_bert_cfg.feature_num, norm_acc=norm_acc)]

    separated_data_test, separated_label_test = separate_data_and_labels_by_user(data_test, label_test_full[:, 0, :])
    test_dataloaders = [] 
    for ind, data_set in enumerate(separated_data_test):  
        data_set_test = IMUDataset(data_set, separated_label_test[ind][:,args.dataset_cfg.activity_label_index], pipeline=pipeline)
        data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size)
        test_dataloaders.append(data_loader_test)

    print("testing data shape", data_test.shape)
    print("testing label shape", label_test.shape)

    data_set_test = IMUDataset(data_test, label_test, pipeline=pipeline)
    data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size)

    classifier = fetch_classifier(method, model_classifier_cfg, input=model_bert_cfg.hidden, output=label_num)
    print(classifier)
    model = BERTClassifier(model_bert_cfg, classifier=classifier, frozen_bert=frozen_bert)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)
    trainer = train.Trainer(train_cfg, model, optimizer, args.save_path, get_device(args.gpu))

    def func_forward(model, batch):
        inputs, label = batch
        logits = model(inputs, False)
        return logits, label

    # For evaluation
    label_estimate_test = trainer.run(func_forward, None, data_loader_test, model_file=args.pretrain_model, load_self=True)
    
    
    return label_test, label_estimate_test


if __name__ == "__main__":
    train_rate = 0.8
    label_rate = 1.0
    balance = True
    balance_ratio = 500
    frozen_bert = True
    method = "base_gru"
    args = handle_argv('evaluate_bert_' + method, 'bert_classifier_train.json', method)
    if args.label_index != -1:
        label_index = args.label_index
    label_names, label_num = load_dataset_label_names(args.dataset_cfg, args.label_index)
    print(label_names)
    label_test, label_estimate_test = evaluate(args, args.label_index, train_rate, label_rate
                                                    , frozen_bert=frozen_bert, balance=balance, balance_ratio=balance_ratio)


    
    acc, matrix, f1 = stat_results(label_test, label_estimate_test)
    matrix_norm = plot_matrix(matrix, label_names)