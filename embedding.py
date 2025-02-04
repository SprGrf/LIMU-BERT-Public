# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 13/1/2021
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : embedding.py
# @Description : generate embeddings using pretrained LIMU-BERT models
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import train
from config import load_dataset_label_names
from models import LIMUBertModel4Pretrain
from plot import plot_reconstruct_sensor, plot_embedding
from utils import LIBERTDataset4Pretrain, load_pretrain_data_config, load_pretrain_config, get_device, handle_argv, \
    Preprocess4Normalization, IMUDataset, prepare_datasets_participants


def fetch_setup(args, output_embed):
    original = False

    if original:
        data, labels, train_cfg, model_cfg, mask_cfg, dataset_cfg = load_pretrain_data_config(args)
    else:
        train_cfg, model_cfg, mask_cfg, dataset_cfg = load_pretrain_config(args)

    pipeline = [Preprocess4Normalization(model_cfg.feature_num)]
    
    if original:
        pass    
    else:
        data_train, labels_train, data_val, labels_val, data_test, labels_test = prepare_datasets_participants(args, seed=train_cfg.seed)

    data_set_train = IMUDataset(data_train, labels_train, pipeline=pipeline)
    data_set_valid = IMUDataset(data_val, labels_val, pipeline=pipeline)
    data_set_test = IMUDataset(data_test, labels_test, pipeline=pipeline)
    data_loader_train = DataLoader(data_set_train, shuffle=False, batch_size=train_cfg.batch_size)
    data_loader_valid = DataLoader(data_set_valid, shuffle=False, batch_size=train_cfg.batch_size)
    data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size)
    model = LIMUBertModel4Pretrain(model_cfg, output_embed=output_embed)
    criterion = nn.MSELoss(reduction='none')
    return data_loader_train, data_loader_valid, data_loader_test, model, criterion, train_cfg


def generate_embedding_or_output(args, save=False, output_embed=True):
    data_loader_train, data_loader_valid, data_loader_test, model, criterion, train_cfg \
        = fetch_setup(args, output_embed)

    optimizer = None
    trainer = train.Trainer(train_cfg, model, optimizer, args.save_path, get_device(args.gpu))
    # trainer = train.Trainer(train_cfg, model, optimizer, args.save_path, 'cpu')

    def func_forward(model, batch):
        seqs, label = batch
        embed = model(seqs)
        return embed, label

    save_name = 'embed_' + args.model_file.split('.')[0] + '_' + args.dataset + '_' + args.dataset_version

    trainer.run_mem(func_forward, None, data_loader_train, args.pretrain_model, name=save_name + "_train")
    # trainer.run_mem(func_forward, None, data_loader_valid, args.pretrain_model, name=save_name + "_valid")
    # trainer.run_mem(func_forward, None, data_loader_test, args.pretrain_model, name=save_name + "_test")

    return

def load_embedding_label(model_file, dataset, dataset_version):
    embed_name = 'embed_' + model_file + '_' + dataset + '_' + dataset_version
    label_name = 'label_' + dataset_version
    embed = np.load(os.path.join('embed', embed_name + '.npy')).astype(np.float32)
    labels = np.load(os.path.join('dataset', dataset, label_name + '.npy')).astype(np.float32)
    return embed, labels

def load_part_file(filename):
    loaded_results, loaded_labels = torch.load(filename, weights_only=False)  
    # print(type(loaded_labels))
    # print(type(loaded_results))

    return loaded_results, loaded_labels

if __name__ == "__main__":
    save = True
    mode = "base"
    args = handle_argv('pretrain_' + mode, 'pretrain.json', mode)
    generate_embedding_or_output(args=args, output_embed=True, save=save)

    # label_index = 1
    # label_names, label_num = load_dataset_label_names(args.dataset_cfg, label_index)
    # data_tsne, labels_tsne = plot_embedding(output, labels, label_index=label_index, reduce=1000, label_names=label_names)
