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


def bert_classify(args, label_index, training_rate, label_rate, frozen_bert=False, balance=True, balance_ratio=100):
    wandb.init(project='classifier', entity='spgaryf')
    wandb.config.balance = balance
    wandb.config.balance_ratio = balance_ratio
    wandb.config.method = method

    if args.dataset != 'c24':
        data, labels, train_cfg, model_bert_cfg, model_classifier_cfg, dataset_cfg = load_bert_classifier_data_config(args)
    else:
        train_cfg, model_bert_cfg, model_classifier_cfg, dataset_cfg = load_bert_classifier_config(args)
    
    label_names, label_num = load_dataset_label_names(dataset_cfg, label_index)

    if args.dataset != 'c24':
        data_train, label_train, data_vali, label_vali, data_test, label_test \
            = prepare_classifier_dataset(data, labels, label_index=label_index, training_rate=training_rate,
                                        label_rate=label_rate, merge=model_classifier_cfg.seq_len, seed=train_cfg.seed
                                        , balance=balance)    
    else:
        data_train, label_train, data_vali, label_vali, data_test, label_test_full = prepare_datasets_participants(args, training_rate, seed=train_cfg.seed)
        if balance:
            data_train, label_train = balance_dataset(data_train, label_train, balance_ratio)

    label_test = label_test_full[:, 0, args.dataset_cfg.activity_label_index]    
 
    norm_acc = False if args.dataset == 'c24' else True
    pipeline = [Preprocess4Normalization(model_bert_cfg.feature_num, norm_acc=norm_acc)]

    separated_data_test, separated_label_test = separate_data_and_labels_by_user(data_test, label_test_full[:, 0, :])
    test_dataloaders = []
    for ind, data_set in enumerate(separated_data_test):  
        data_set_test = IMUDataset(data_set, separated_label_test[ind][:,args.dataset_cfg.activity_label_index], pipeline=pipeline)
        data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size)
        test_dataloaders.append(data_loader_test)

    print("training data shape", data_train.shape)
    print("training label shape", label_train.shape)
    print("validation data shape", data_vali.shape)
    print("validation label shape", label_vali.shape)
    print("testing data shape", data_test.shape)
    print("testing label shape", label_test.shape)



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

    

    # data_set_train = IMUDataset(data_train, label_train, pipeline=pipeline)
    # data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size)
    data_set_test = IMUDataset(data_test, label_test, pipeline=pipeline)
    data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size)
    data_set_vali = IMUDataset(data_vali, label_vali, pipeline=pipeline)
    data_loader_vali = DataLoader(data_set_vali, shuffle=False, batch_size=train_cfg.batch_size)


    # # Weighted loss part
    # class_weights = compute_class_weight('balanced', classes=np.unique(label_train), y=label_train)
    # class_weights = torch.tensor(class_weights, dtype=torch.float32).to(get_device(args.gpu))
    # criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Normal loss
    criterion = nn.CrossEntropyLoss()

    classifier = fetch_classifier(method, model_classifier_cfg, input=model_bert_cfg.hidden, output=label_num)
    print(classifier)
    model = BERTClassifier(model_bert_cfg, classifier=classifier, frozen_bert=frozen_bert)
    # print(model)      
    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)
    trainer = train.Trainer(train_cfg, model, optimizer, args.save_path, get_device(args.gpu))
    if method == 'base_gru':
        wandb.config.num_rnn = model_classifier_cfg.num_rnn
        wandb.config.num_linear = model_classifier_cfg.num_linear
        wandb.config.seq_len = model_classifier_cfg.seq_len
        wandb.config.num_layers = model_classifier_cfg.num_layers
    
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
    trainer.train(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test,  data_loader_vali
                        , test_dataloaders, model_file=args.pretrain_model, load_self=True)
    label_estimate_test = trainer.run(func_forward, None, data_loader_test)
    
    
    ## For evaluation
    # label_estimate_test = trainer.run(func_forward, None, data_loader_test, model_file=args.pretrain_model, load_self=True)
    
    
    return label_test, label_estimate_test


if __name__ == "__main__":
    train_rate = 0.8
    label_rate = 1.0
    balance = True
    balance_ratio = 500
    frozen_bert = True
    method = "base_gru"
    # args = handle_argv('bert_classifier_' + method, 'bert_classifier_train.json', method)
    args = handle_argv('evaluate_bert_' + method, 'bert_classifier_train.json', method)
    if args.label_index != -1:
        label_index = args.label_index
    label_names, label_num = load_dataset_label_names(args.dataset_cfg, args.label_index)
    print(label_names)
    label_test, label_estimate_test = bert_classify(args, args.label_index, train_rate, label_rate
                                                    , frozen_bert=frozen_bert, balance=balance, balance_ratio=balance_ratio)


    
    acc, matrix, f1 = stat_results(label_test, label_estimate_test)
    matrix_norm = plot_matrix(matrix, label_names)