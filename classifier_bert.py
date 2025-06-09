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
from utils import set_seeds, get_device,  handle_argv, get_sample_weights, separate_data_and_labels_by_user, \
    IMUDataset, load_bert_classifier_data_config, load_bert_classifier_config, Preprocess4Normalization, \
    prepare_classifier_dataset, prepare_datasets_participants, balance_dataset


def bert_classify(args, label_index, training_rate, rnd=None, frozen_bert=False, balance=True, balance_ratio=100, loocv=False):
    train_cfg, model_bert_cfg, model_classifier_cfg, dataset_cfg = load_bert_classifier_config(args)
    
    label_names, label_num = load_dataset_label_names(dataset_cfg, label_index)

    data_train, label_train, data_vali, label_vali, data_test, label_test_full = prepare_datasets_participants(args, training_rate, seed=train_cfg.seed, loocv=loocv, round=rnd)
    if balance:
        data_train, label_train = balance_dataset(data_train, label_train, balance_ratio)

    label_test = label_test_full[:, 0, args.dataset_cfg.activity_label_index]    
 
    # norm_acc = False if args.dataset == 'C24' else True
    norm_acc = False
    pipeline = [Preprocess4Normalization(model_bert_cfg.feature_num, norm_acc=norm_acc)]

    separated_data_test, separated_label_test = separate_data_and_labels_by_user(data_test, label_test_full[:, 0, :])
    test_dataloaders = []
    for ind, data_set in enumerate(separated_data_test):  
        data_set_test = IMUDataset(data_set, separated_label_test[ind][:,args.dataset_cfg.activity_label_index], pipeline=pipeline)
        data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size)
        test_dataloaders.append(data_loader_test)

    ## Sampler dataloader
    unique_ytrain, counts_ytrain = np.unique(label_train, return_counts=True)
    print('y_train label distribution: ', dict(zip(unique_ytrain, counts_ytrain)))
    weights = 100.0 / torch.Tensor(counts_ytrain)
    weights = weights.double()
    print('weights of sampler: ', weights)
    sample_weights = get_sample_weights(label_train, weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    data_set_train = IMUDataset(data_train, label_train, pipeline=pipeline)
    data_loader_train = DataLoader(data_set_train, shuffle=False, batch_size=train_cfg.batch_size, sampler=sampler)

    

    data_set_test = IMUDataset(data_test, label_test, pipeline=pipeline)
    data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size)
    data_set_vali = IMUDataset(data_vali, label_vali, pipeline=pipeline)
    data_loader_vali = DataLoader(data_set_vali, shuffle=False, batch_size=train_cfg.batch_size)


    # ## Weighted loss part
    # class_weights = compute_class_weight('balanced', classes=np.unique(label_train), y=label_train)
    # class_weights = torch.tensor(class_weights, dtype=torch.float32).to(get_device(args.gpu))
    # print("class weights are: ", class_weights)
    # criterion = nn.CrossEntropyLoss(weight=class_weights)

    ## Normal loss
    criterion = nn.CrossEntropyLoss()

    classifier = fetch_classifier(method, model_classifier_cfg, input=model_bert_cfg.hidden, output=label_num)
    model = BERTClassifier(model_bert_cfg, classifier=classifier, frozen_bert=frozen_bert)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)
    trainer = train.Trainer(train_cfg, model, optimizer, args.save_path + "_round_" + str(rnd), get_device(args.gpu))

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

    ## For training
    if rnd != None and not args.complex:
        pretrained_encoder_path = args.pretrain_model + "_round_" + str(rnd)
    else: # if we have a complex training case or on C24
        pretrained_encoder_path = args.pretrain_model
    trainer.train(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test,  data_loader_vali
                        , test_dataloaders, model_file=pretrained_encoder_path, load_self=True)
    label_estimate_test = trainer.run(func_forward, None, data_loader_test)
    
    
    # For evaluation
    label_estimate_test = trainer.run(func_forward, None, data_loader_test, model_file=args.save_path + "_round_" + str(rnd), load_self=True)
    

    return label_test, label_estimate_test


if __name__ == "__main__":
    train_rate = 0.8
    balance = False
    balance_ratio = 10
    frozen_bert = True
    method = "base_gru"
    args = handle_argv('bert_classifier_' + method, 'bert_classifier_train.json', method)
    if args.label_index != -1:
        label_index = args.label_index
    label_names, label_num = load_dataset_label_names(args.dataset_cfg, args.label_index)
    print(label_names)
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
            # args.save_path += "_round_" + str(round)
            label_test, label_estimate_test = bert_classify(args, args.label_index, train_rate, rnd=round
                                                    , frozen_bert=frozen_bert, balance=balance, balance_ratio=balance_ratio, loocv=loocv)


    
        acc, matrix, f1 = stat_results(label_test, label_estimate_test)
        matrix_norm = plot_matrix(matrix, label_names)