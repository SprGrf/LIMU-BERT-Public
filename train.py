#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/16 11:22
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : train.py
# @Description :
import copy
import os
import time
from tqdm import tqdm
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import count_model_parameters, find_files, prepare_classifier_dataset, load_classifier_config, set_seeds, IMUDataset
from embedding import load_part_file


class Trainer(object):
    """Training Helper Class"""
    def __init__(self, cfg, model, optimizer, save_path, device):
        self.cfg = cfg # config for training : see class Config
        self.model = model
        self.optimizer = optimizer
        self.save_path = save_path
        self.device = device # device name

    def pretrain(self, func_loss, func_forward, func_evaluate
              , data_loader_train, data_loader_test, model_file=None, data_parallel=False):
        """ Train Loop """
        self.load(model_file)
        model = self.model.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        global_step = 0 # global iteration steps regardless of epochs
        best_loss = 1e6
        model_best = model.state_dict()

        for e in range(self.cfg.n_epochs):
            loss_sum = 0. # the sum of iteration losses to get average loss in every epoch
            time_sum = 0.0
            self.model.train()
            for i, batch in enumerate(data_loader_train):
                batch = [t.to(self.device) for t in batch]
                start_time = time.time()
                self.optimizer.zero_grad()
                loss = func_loss(model, batch)

                loss = loss.mean()# mean() for Data Parallelism
                loss.backward()
                self.optimizer.step()
                time_sum += time.time() - start_time
                
                global_step += 1
                loss_sum += loss.item()

                # if global_step % self.cfg.save_steps == 0: # save
                #     self.save(global_step)

                if self.cfg.total_steps and self.cfg.total_steps < global_step:
                    print('The Total Steps have been reached.')
                    return
                # print(i)

            loss_eva = self.run(func_forward, func_evaluate, data_loader_test)
            print('Epoch %d/%d : Average Loss %5.4f. Test Loss %5.4f'
                    % (e + 1, self.cfg.n_epochs, loss_sum / len(data_loader_train), loss_eva))
            # print("Train execution time: %.5f seconds" % (time_sum / len(data_loader_train)))
            if loss_eva < best_loss:
                best_loss = loss_eva
                model_best = copy.deepcopy(model.state_dict())
                self.save(0)
        model.load_state_dict(model_best)
        print('The Total Epoch have been reached.')
        # self.save(global_step)

    # def run(self, func_forward, func_evaluate, data_loader, model_file=None, data_parallel=False, load_self=False, save_path=None):
    #     """ Evaluation Loop """
    #     self.model.eval() # evaluation mode
    #     self.load(model_file, load_self=load_self)
    #     # print(count_model_parameters(self.model))
    #     model = self.model.to(self.device)
    #     if data_parallel: # use Data Parallelism with Multi-GPU
    #         model = nn.DataParallel(model)


    #     results_fp = open(save_path, "ab")
        
    #     time_start = time.time()

    #     for batch in data_loader:
    #         batch = [t.to(self.device) for t in batch]
    #         with torch.no_grad():
    #             start_time = time.time()
    #             result, label = func_forward(model, batch)

    #             # Move to CPU and convert to NumPy immediately
    #             result_np = result.cpu().numpy()
    #             label_np = label.cpu().numpy()

    #             # Save batch directly to disk (append mode)
    #             np.save(results_fp, result_np)

    #         # Clear memory
    #         del result, label, result_np, label_np
    #         torch.cuda.empty_cache()

    #     print("Eval execution time: %.5f seconds" % (time.time() - time_start ))
    #     if func_evaluate:
    #         pass
    #         # return func_evaluate(torch.cat(labels, 0), torch.cat(results, 0))
    #     else:
    #         # Close file pointers
    #         print("saving")
    #         results_fp.close()
    #         return 

    def run(self, func_forward, func_evaluate, data_loader, model_file=None, data_parallel=False, load_self=False):
        """ Evaluation Loop """
        self.model.eval() # evaluation mode
        self.load(model_file, load_self=load_self)
        # print(count_model_parameters(self.model))
        model = self.model.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        results = [] # prediction results
        labels = []
        time_sum = 0.0
        for batch in data_loader:
            batch = [t.to(self.device) for t in batch]
            with torch.no_grad(): # evaluation without gradient calculation
                start_time = time.time()
                result, label = func_forward(model, batch)
                time_sum += time.time() - start_time
                results.append(result)
                labels.append(label)
        # print("Eval execution time: %.5f seconds" % (time_sum / len(dt)))
        if func_evaluate:
            return func_evaluate(torch.cat(labels, 0), torch.cat(results, 0))
        else:
            return torch.cat(results, 0).cpu().numpy()
        

    def run_mem(self, func_forward, func_evaluate, data_loader, model_file=None, 
            data_parallel=False, load_self=False, memory_threshold=2e9, name=''):
        """ Evaluation Loop with Memory Management """
        
        self.model.eval()  # Evaluation mode
        self.load(model_file, load_self=load_self)
        model = self.model.to(self.device)

        if data_parallel:  # Use Data Parallelism with Multi-GPU
            model = torch.nn.DataParallel(model)

        results = []  # Prediction results
        labels = []
        part_counter = 0  # Track file index
        start_time = time.time()

        for batch in data_loader:
            batch = [t.to(self.device) for t in batch]
            with torch.no_grad():  # Evaluation without gradient calculation
                start_time = time.time()
                result, label = func_forward(model, batch)
                results.append(result)
                labels.append(label)

            # Check memory usage and dump results if needed
            if sum(t.numel() * t.element_size() for t in results) > memory_threshold:
                print("saving file", part_counter)
                part_file_embed = f"embed/{name}_part_{part_counter}.pt"
                torch.save((torch.cat(results, 0).cpu().numpy(), torch.cat(labels, 0).cpu().numpy()), part_file_embed)
                part_counter += 1
                
                # Free memory
                del results, labels
                results, labels = [], []
                torch.cuda.empty_cache()

        # Save remaining results if needed
        if results:
            print("saving file", part_counter)
            part_file_embed = f"embed/{name}_part_{part_counter}.pt"
            torch.save((torch.cat(results, 0).cpu().numpy(), torch.cat(labels, 0).cpu().numpy()), part_file_embed)
            # Free memory
            del results, labels
            results, labels = [], []
            torch.cuda.empty_cache()
        print("Eval execution time: %.5f seconds" % (time.time() - start_time ))


    def train(self, func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test, data_loader_vali
              , model_file=None, data_parallel=False, load_self=False):
        """ Train Loop """
        self.load(model_file, load_self)
        model = self.model.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        global_step = 0 # global iteration steps regardless of epochs
        vali_acc_best = 0.0
        best_stat = None
        model_best = model.state_dict()
        for e in range(self.cfg.n_epochs):
            loss_sum = 0.0 # the sum of iteration losses to get average loss in every epoch
            time_sum = 0.0
            self.model.train()
            for i, batch in enumerate(data_loader_train):
                batch = [t.to(self.device) for t in batch]

                start_time = time.time()
                self.optimizer.zero_grad()
                loss = func_loss(model, batch)

                loss = loss.mean()# mean() for Data Parallelism
                loss.backward()
                self.optimizer.step()

                global_step += 1
                loss_sum += loss.item()
                time_sum += time.time() - start_time
                if self.cfg.total_steps and self.cfg.total_steps < global_step:
                    print('The Total Steps have been reached.')
                    return
            train_acc, train_f1, train_mr = self.run(func_forward, func_evaluate, data_loader_train)
            test_acc, test_f1, test_mr = self.run(func_forward, func_evaluate, data_loader_test)
            vali_acc, vali_f1, vali_mr = self.run(func_forward, func_evaluate, data_loader_vali)
            print('Epoch %d/%d : Average Loss %5.4f, Accuracy: %0.3f/%0.3f/%0.3f, F1: %0.3f/%0.3f/%0.3f, Mean Recall: %0.3f/%0.3f/%0.3f'
                  % (e+1, self.cfg.n_epochs, loss_sum / len(data_loader_train), train_acc, vali_acc, test_acc, train_f1, vali_f1, test_f1, train_mr, vali_mr, test_mr))
            # print("Train execution time: %.5f seconds" % (time_sum / len(self.data_loader)))
            if vali_acc > vali_acc_best:
                vali_acc_best = vali_acc
                best_stat = (train_acc, vali_acc, test_acc, train_f1, vali_f1, test_f1)
                model_best = copy.deepcopy(model.state_dict())
                self.save(e)
        self.model.load_state_dict(model_best)
        print('The Total Epoch have been reached.')
        print('Best Accuracy: %0.3f/%0.3f/%0.3f, F1: %0.3f/%0.3f/%0.3f' % best_stat)


    def train_parts(self, args, func_loss, func_forward, func_evaluate, label_rate, label_index, balance, 
              model_file=None, data_parallel=False, load_self=False):
        
        
        """ Train Loop """
        self.load(model_file, load_self)
        model = self.model.to(self.device)
        train_cfg, model_cfg, dataset_cfg = load_classifier_config(args)
        set_seeds(train_cfg.seed)

        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        global_step = 0 # global iteration steps regardless of epochs
        vali_acc_best = 0.0
        best_stat = None
        model_best = model.state_dict()


        training_embeding_files = find_files('embed', 'train_part')
        # print("training embeddings",training_embeding_files)
        validation_embeding_files = find_files('embed', 'valid_part')
        # print("validation embeddings",validation_embeding_files)
        testing_embeding_files = find_files('embed', 'test_part')
        # print("testing embeddings", testing_embeding_files)
        
        
        # ## Validation dataloader
        # data_valid = []
        # label_valid = []
        # for valid_file in validation_embeding_files:
        #     print(valid_file)
        #     ev, lv = load_part_file(valid_file) 
        #     data_valid.append(ev)  
        #     label_valid.append(lv) 
        # data_valid = np.concatenate(data_valid, axis=0)
        # label_valid = np.concatenate(label_valid, axis=0)

        # data_valid, label_valid, _, _, _, _ \
        #         = prepare_classifier_dataset(data_valid, label_valid, label_index=label_index, training_rate=1.0
        #                                         , label_rate=label_rate, merge=model_cfg.seq_len, seed=train_cfg.seed
        #                                         , balance=balance)

        # data_set_valid = IMUDataset(data_valid, label_valid)
        # del data_valid, label_valid, ev, lv
        # data_loader_valid = DataLoader(data_set_valid, shuffle=True, batch_size=train_cfg.batch_size)
        # del data_set_valid
        
        # ## Testing dataloader
        # data_test = []
        # label_test = []
        # for test_file in testing_embeding_files:
        #     print(test_file)
        #     et, lt = load_part_file(test_file) 
        #     data_test.append(et)  
        #     label_test.append(lt) 
        # data_test = np.concatenate(data_test, axis=0)
        # label_test = np.concatenate(label_test, axis=0)

        # data_test, label_test, _, _, _, _ \
        #         = prepare_classifier_dataset(data_test, label_test, label_index=label_index, training_rate=1.0
        #                                         , label_rate=label_rate, merge=model_cfg.seq_len, seed=train_cfg.seed
        #                                         , balance=balance)

        # data_set_test = IMUDataset(data_test, label_test)
        # data_loader_test = DataLoader(data_set_test, shuffle=True, batch_size=train_cfg.batch_size)
        # del data_set_test, data_test, et, lt #, label_test



        # for e in range(self.cfg.n_epochs):
        #     loss_sum = 0.0 # the sum of iteration losses to get average loss in every epoch
        #     time_sum = 0.0


        #     for i in tqdm(range(len(training_embeding_files)), desc="Processing Files", unit="file", leave="True"):
        #     # for file_no, file_name in enumerate(training_embeding_files):
        #         self.model.train()

        #         # print("file number ", file_no)
        #         embedding, labels = load_part_file(training_embeding_files[i])
        #         # #######################################################3
        #         data_train, label_train, _, _, _, _ \
        #             = prepare_classifier_dataset(embedding, labels, label_index=label_index, training_rate=1.0
        #                                          , label_rate=label_rate, merge=model_cfg.seq_len, seed=train_cfg.seed
        #                                          , balance=balance)
                
        batch_size = 6  # Number of files to load at once
        num_files = len(training_embeding_files)
        
        # for e in range(self.cfg.n_epochs):
        for e in range(1):
            loss_sum = 0.0  # Sum of iteration losses to get the average loss per epoch
            time_sum = 0.0
            
            random.shuffle(training_embeding_files)
            data_loader_length = 0
            for i in tqdm(range(0, num_files, batch_size), desc="Processing Files", unit="batch", leave=True):
                self.model.train()

                # Select a batch of files
                batch_files = training_embeding_files[i:i + batch_size]

                embeddings, labels = [], []
                for file_name in batch_files:
                    emb, lbl = load_part_file(file_name)  # Load each file
                    embeddings.append(emb)
                    labels.append(lbl)

                del emb, lbl
                # Merge embeddings and labels from multiple files
                embeddings = np.concatenate(embeddings, axis=0)
                labels = np.concatenate(labels, axis=0)

                # Prepare dataset using the batch
                data_train, label_train, _, _, _, _ = prepare_classifier_dataset(
                    embeddings, labels, label_index=label_index, training_rate=1.0,
                    label_rate=label_rate, merge=model_cfg.seq_len, seed=train_cfg.seed,
                    balance=balance
                )
                
                del embeddings, labels
                data_set_train = IMUDataset(data_train, label_train)
                del data_train, label_train
                data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size)\
                # data_loader_train = DataLoader(data_set_train, batch_size=train_cfg.batch_size, shuffle=True, 
                #                                 pin_memory=True, num_workers=4)
                del data_set_train
                
                # unique_label_train, counts_train = np.unique(label_train, return_counts=True)
                # unique_label_vali, counts_vali = np.unique(label_vali, return_counts=True)
                # unique_label_test, counts_test = np.unique(label_test, return_counts=True)
                # print('Train label distribution: ', dict(zip(unique_label_train, counts_train)))
                # print('Validation label distribution: ', dict(zip(unique_label_vali, counts_vali)))
                # print('Test label distribution: ', dict(zip(unique_label_test, counts_test)))
                # ##############################################################
                data_loader_length += len(data_loader_train)
                # print(data_loader_length)
                for i, batch in enumerate(data_loader_train):
                    batch = [t.to(self.device) for t in batch]

                    start_time = time.time()
                    self.optimizer.zero_grad()
                    loss = func_loss(model, batch)

                    loss = loss.mean()# mean() for Data Parallelism
                    loss.backward()
                    self.optimizer.step()

                    global_step += 1
                    loss_sum += loss.item()
                    time_sum += time.time() - start_time
                    if self.cfg.total_steps and self.cfg.total_steps < global_step:
                        print('The Total Steps have been reached.')
                        return
                train_acc, train_f1, train_mr = self.run(func_forward, func_evaluate, data_loader_train)
                del data_loader_train

            ############################################################################
            ############################################################################
            ############################################################################

            ## Validation dataloader
            data_valid = []
            label_valid = []
            for valid_file in validation_embeding_files:
                print(valid_file)
                ev, lv = load_part_file(valid_file) 
                data_valid.append(ev)  
                label_valid.append(lv) 
            data_valid = np.concatenate(data_valid, axis=0)
            label_valid = np.concatenate(label_valid, axis=0)

            data_valid, label_valid, _, _, _, _ \
                    = prepare_classifier_dataset(data_valid, label_valid, label_index=label_index, training_rate=1.0
                                                    , label_rate=label_rate, merge=model_cfg.seq_len, seed=train_cfg.seed
                                                    , balance=balance)

            data_set_valid = IMUDataset(data_valid, label_valid)
            del data_valid, label_valid, ev, lv
            data_loader_valid = DataLoader(data_set_valid, shuffle=True, batch_size=train_cfg.batch_size)
            del data_set_valid
            
            vali_acc, vali_f1, vali_mr = self.run(func_forward, func_evaluate, data_loader_valid)
            
            del data_loader_valid

            ## Testing dataloader
            data_test = []
            label_test = []
            for test_file in testing_embeding_files:
                print(test_file)
                et, lt = load_part_file(test_file) 
                data_test.append(et)  
                label_test.append(lt) 
            data_test = np.concatenate(data_test, axis=0)
            label_test = np.concatenate(label_test, axis=0)

            data_test, label_test, _, _, _, _ \
                    = prepare_classifier_dataset(data_test, label_test, label_index=label_index, training_rate=1.0
                                                    , label_rate=label_rate, merge=model_cfg.seq_len, seed=train_cfg.seed
                                                    , balance=balance)

            data_set_test = IMUDataset(data_test, label_test)
            del data_test, et, lt #, label_test
            data_loader_test = DataLoader(data_set_test, shuffle=True, batch_size=train_cfg.batch_size)
            del data_set_test
            test_acc, test_f1, test_mr = self.run(func_forward, func_evaluate, data_loader_test)
            # del data_loader_test

            ############################################################################
            ############################################################################
            ############################################################################
            ############################################################################

            # print("final", data_loader_length)
            # test_acc, test_f1, test_mr = self.run(func_forward, func_evaluate, data_loader_test)
            # vali_acc, vali_f1, vali_mr = self.run(func_forward, func_evaluate, data_loader_valid)
            print('Epoch %d/%d : Average Loss %5.4f, Accuracy: %0.3f/%0.3f/%0.3f, F1: %0.3f/%0.3f/%0.3f, Mean Recall: %0.3f/%0.3f/%0.3f'
                % (e+1, self.cfg.n_epochs, loss_sum / data_loader_length, train_acc, vali_acc, test_acc, train_f1, vali_f1, test_f1, train_mr, vali_mr, test_mr))
            # print("Train execution time: %.5f seconds" % (time_sum / len(self.data_loader)))
            if vali_acc > vali_acc_best:
                vali_acc_best = vali_acc
                best_stat = (train_acc, vali_acc, test_acc, train_f1, vali_f1, test_f1)
                model_best = copy.deepcopy(model.state_dict())
                self.save(e)

        self.model.load_state_dict(model_best)
        print('The Total Epoch have been reached.')
        print('Best Accuracy: %0.3f/%0.3f/%0.3f, F1: %0.3f/%0.3f/%0.3f' % best_stat)
        
        label_estimate_test = self.run(func_forward, None, data_loader_test)
        return  label_test, label_estimate_test


    def load(self, model_file, load_self=False):
        """ load saved model or pretrained transformer (a part of model) """
        if model_file:
            print('Loading the model from', model_file)
            if load_self:
                self.model.load_self(model_file + '.pt', map_location=self.device)
            else:
                self.model.load_state_dict(torch.load(model_file + '.pt', map_location=self.device))

    def save(self, i=0):
        """ save current model """
        print("Saving model")
        if i != 0:
            torch.save(self.model.state_dict(), self.save_path + "_" + str(i) + '.pt')
        else:
            torch.save(self.model.state_dict(),  self.save_path + '.pt')

