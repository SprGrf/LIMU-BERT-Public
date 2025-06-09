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
from sklearn.metrics import confusion_matrix
import train
import os
from config import load_dataset_label_names
from models import BERTClassifier, fetch_classifier
from plot import plot_matrix
import pickle as cp
from config import load_dataset_stats
from statistic import stat_acc_f1_rec, stat_results
from utils import set_seeds, get_device,  handle_argv, separate_data_and_labels_by_user, \
    IMUDataset, load_bert_classifier_config, Preprocess4Normalization, \
    prepare_datasets_participants


def evaluate(args, label_index, training_rate, frozen_bert=False, loocv=False, round=None):

    train_cfg, model_bert_cfg, model_classifier_cfg, dataset_cfg = load_bert_classifier_config(args)
    
    label_names, label_num = load_dataset_label_names(dataset_cfg, label_index)
    print("dataset is", args.dataset)
    _, _, _, _, data_test, label_test_full = prepare_datasets_participants(args, training_rate, seed=train_cfg.seed, loocv=loocv, round=round)
    
    label_test = label_test_full[:, 0, args.dataset_cfg.activity_label_index]     
    
    norm_acc = False
    pipeline = [Preprocess4Normalization(model_bert_cfg.feature_num, norm_acc=norm_acc)]

    separated_data_test, separated_label_test = separate_data_and_labels_by_user(data_test, label_test_full[:, 0, :])
    test_dataloaders = [] 
    true_labels = []
    for ind, data_set in enumerate(separated_data_test):  
        true_labels.append(separated_label_test[ind][:,args.dataset_cfg.activity_label_index])
        data_set_test_single = IMUDataset(data_set, separated_label_test[ind][:,args.dataset_cfg.activity_label_index], pipeline=pipeline)
        data_loader_test_single = DataLoader(data_set_test_single, shuffle=False, batch_size=train_cfg.batch_size)
        test_dataloaders.append(data_loader_test_single)


    classifier = fetch_classifier(method, model_classifier_cfg, input=model_bert_cfg.hidden, output=label_num)
    model = BERTClassifier(model_bert_cfg, classifier=classifier, frozen_bert=frozen_bert)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)
    trainer = train.Trainer(train_cfg, model, optimizer, args.save_path, get_device(args.gpu))

    def func_forward(model, batch):
        inputs, label = batch
        logits = model(inputs, False)
        return logits, label

    # For evaluation
    if round != None:
        pretrained_model = args.pretrain_model + "_round_" + str(round)
    else:
        pretrained_model = args.pretrain_model + "_round_" + str('0') ## TODO change this, just for tests 
    print("model to load is", pretrained_model)
    estimate_labels = []
    for td in test_dataloaders:
        label_estimate_test = trainer.run(func_forward, None, td, model_file=pretrained_model, load_self=True)
        estimate_labels.append(label_estimate_test)
    
    return true_labels, estimate_labels


if __name__ == "__main__":
    train_rate = 0.8
    frozen_bert = True
    method = "base_gru"
    args = handle_argv('evaluate_bert_' + method, 'bert_classifier_train.json', method)
    if args.label_index != -1:
        label_index = args.label_index
    label_names, label_num = load_dataset_label_names(args.dataset_cfg, args.label_index)
    print(label_names)
    all_datasets = ['HHAR', 'DSA', 'MHEALTH', 'selfBACK', 'PAMAP2', 'GOTOV', 'C24']


    set_seeds(10)
    if args.case_study == "cv":
        cm_round_filename = os.path.join("saved","evaluation_results", args.case_study, args.dataset)
        os.makedirs(cm_round_filename, exist_ok=True)
        cms =[]
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
            labels, labels_estimate = evaluate(args, args.label_index, train_rate, frozen_bert=frozen_bert, loocv=loocv, round=round)
            for ind, labels_user in enumerate(labels_estimate):
                label_estimated = np.argmax(labels_user, 1)
                conf_matrix = confusion_matrix(labels[ind], label_estimated)
                cms.append(conf_matrix)
        f = open(os.path.join(cm_round_filename, "self.cms"), 'wb')
        cp.dump(cms, f, protocol=cp.HIGHEST_PROTOCOL)
        f.close()
    elif args.case_study == "d2d_test":  
        if args.dataset != 'C24':
            train_dataset = args.dataset
            # in 2 in
            test_datasets = [ds for ds in all_datasets if ds != args.dataset and ds != 'C24']
            cm_round_filename = os.path.join("saved","evaluation_results", args.case_study, args.dataset)
            os.makedirs(cm_round_filename, exist_ok=True)
            cms_in2in =[]
            for ds in test_datasets:
                cms_specific = []
                print("Evaluating on", ds)
                args.dataset = ds
                args.dataset_cfg = load_dataset_stats(args.dataset, args.dataset_version)
                labels, labels_estimate = evaluate(args, args.label_index, train_rate, frozen_bert=frozen_bert)
                for ind, labels_user in enumerate(labels_estimate):
                    label_estimated = np.argmax(labels_user, 1)
                    conf_matrix = confusion_matrix(labels[ind], label_estimated)
                    cms_in2in.append(conf_matrix)
                    cms_specific.append(conf_matrix)

                f_specific = open(os.path.join(cm_round_filename, train_dataset + "_" + ds +  ".cms"), 'wb')
                cp.dump(cms_specific, f_specific, protocol=cp.HIGHEST_PROTOCOL)
                f_specific.close()
            f = open(os.path.join(cm_round_filename, "in2in.cms"), 'wb')
            cp.dump(cms_in2in, f, protocol=cp.HIGHEST_PROTOCOL)
            f.close()
            #### in 2 out 
            ds = 'C24'
            cms_in2out =[]
            print("Evaluating on", ds)
            args.dataset = ds
            args.dataset_cfg = load_dataset_stats(args.dataset, args.dataset_version)
            labels, labels_estimate = evaluate(args, args.label_index, train_rate, frozen_bert=frozen_bert)
            for ind, labels_user in enumerate(labels_estimate):
                label_estimated = np.argmax(labels_user, 1)
                conf_matrix = confusion_matrix(labels[ind], label_estimated)
                cms_in2out.append(conf_matrix)
            f = open(os.path.join(cm_round_filename, "in2out.cms"), 'wb')
            cp.dump(cms_in2out, f, protocol=cp.HIGHEST_PROTOCOL)
            f.close()
        else:
            # out 2 in
            test_datasets = [ds for ds in all_datasets if ds != args.dataset]
            cm_round_filename = os.path.join("saved","evaluation_results", args.case_study, args.dataset)
            os.makedirs(cm_round_filename, exist_ok=True)
            cms_in2in =[]
            for ds in test_datasets:
                cms_specific = []
                print("Evaluating on", ds)
                args.dataset = ds
                args.dataset_cfg = load_dataset_stats(args.dataset, args.dataset_version)
                labels, labels_estimate = evaluate(args, args.label_index, train_rate, frozen_bert=frozen_bert)
                for ind, labels_user in enumerate(labels_estimate):
                    label_estimated = np.argmax(labels_user, 1)
                    conf_matrix = confusion_matrix(labels[ind], label_estimated)
                    cms_in2in.append(conf_matrix)
                    cms_specific.append(conf_matrix)

                f_specific = open(os.path.join(cm_round_filename, "C24_" + ds +  ".cms"), 'wb')
                cp.dump(cms_specific, f_specific, protocol=cp.HIGHEST_PROTOCOL)
                f_specific.close()
            f = open(os.path.join(cm_round_filename, "out2in.cms"), 'wb')
            cp.dump(cms_in2in, f, protocol=cp.HIGHEST_PROTOCOL)
            f.close()
            #### out 2 out 
            ds = 'C24'
            cms_in2out =[]
            print("Evaluating on", ds)
            args.dataset = ds
            args.dataset_cfg = load_dataset_stats(args.dataset, args.dataset_version)
            labels, labels_estimate = evaluate(args, args.label_index, train_rate, frozen_bert=frozen_bert)
            for ind, labels_user in enumerate(labels_estimate):
                label_estimated = np.argmax(labels_user, 1)
                conf_matrix = confusion_matrix(labels[ind], label_estimated)
                cms_in2out.append(conf_matrix)
            f = open(os.path.join(cm_round_filename, "out2out.cms"), 'wb')
            cp.dump(cms_in2out, f, protocol=cp.HIGHEST_PROTOCOL)
            f.close()            
