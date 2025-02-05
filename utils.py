#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/16 11:22
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : utils.py
# @Description :

import argparse
import os
from scipy.special import factorial
from torch.utils.data import Dataset

from config import create_io_config, load_dataset_stats, TrainConfig, MaskConfig, load_model_config


""" Utils Functions """

import random
import numpy as np
from collections import Counter
import torch
import sys


def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(gpu):
    "get device (CPU or GPU)"
    if gpu is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:" + gpu if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("%s (%d GPUs)" % (device, n_gpu))


    # device = 'cpu'
    return device


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = x.size(-1) // -np.prod(shape)
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


def bert_mask(seq_len, goal_num_predict):
    return random.sample(range(seq_len), goal_num_predict)


def span_mask(seq_len, max_gram=3, p=0.2, goal_num_predict=15):
    ngrams = np.arange(1, max_gram + 1, dtype=np.int64)
    pvals = p * np.power(1 - p, np.arange(max_gram))
    # alpha = 6
    # pvals = np.power(alpha, ngrams) * np.exp(-alpha) / factorial(ngrams)# possion
    pvals /= pvals.sum(keepdims=True)
    mask_pos = set()
    while len(mask_pos) < goal_num_predict:
        n = np.random.choice(ngrams, p=pvals)
        n = min(n, goal_num_predict - len(mask_pos))
        anchor = np.random.randint(seq_len)
        if anchor in mask_pos:
            continue
        for i in range(anchor, min(anchor + n, seq_len - 1)):
            mask_pos.add(i)
    return list(mask_pos)


def merge_dataset(data, label, mode='all'):
    index = np.zeros(data.shape[0], dtype=bool)
    label_new = []
    for i in range(label.shape[0]):
        if mode == 'all':
            temp_label = np.unique(label[i])
            if temp_label.size == 1:
                index[i] = True
                label_new.append(label[i, 0])
        elif mode == 'any':
            index[i] = True
            if np.any(label[i] > 0):
                temp_label = np.unique(label[i])
                if temp_label.size == 1:
                    label_new.append(temp_label[0])
                else:
                    label_new.append(temp_label[1])
            else:
                label_new.append(0)
        else:
            index[i] = ~index[i]
            label_new.append(label[i, 0])
    # print('Before Merge: %d, After Merge: %d' % (data.shape[0], np.sum(index)))
    return data[index], np.array(label_new)


def reshape_data(data, merge):
    if merge == 0:
        return data.reshape(data.shape[0] * data.shape[1], data.shape[2])
    else:
        # Compute the total number of elements in the first two dimensions
        total_rows = data.shape[0] * data.shape[1]
        truncated_rows = (total_rows // merge) * merge  # Find the largest divisible number

        # if truncated_rows < total_rows:
        #     print(f"Truncating data: from {total_rows} rows to {truncated_rows} rows")

        # Flatten and truncate
        truncated_data = data.reshape(-1, data.shape[2])[:truncated_rows]  

        # Reshape the truncated data
        reshaped_data = truncated_data.reshape(truncated_rows // merge, merge, data.shape[2])

        return reshaped_data

def reshape_label(label, merge):
    if merge == 0:
        return label.reshape(label.shape[0] * label.shape[1])
    else:
        # Compute the total number of elements and adjust for divisibility
        total_rows = label.shape[0] * label.shape[1]
        truncated_rows = (total_rows // merge) * merge  # Find the largest divisible number

        # if truncated_rows < total_rows:
        #     print(f"Truncating labels: from {total_rows} rows to {truncated_rows} rows")

        # Flatten and truncate
        truncated_data = label.reshape(-1)[:truncated_rows]  # Flatten the label data

        # Reshape the truncated data
        reshaped_data = truncated_data.reshape(truncated_rows // merge, merge)

        return reshaped_data


def shuffle_data_label(data, label):
    index = np.arange(data.shape[0])
    np.random.shuffle(index)
    return data[index, ...], label[index, ...]


# def select_participants(users_array, special_participant_list, training_rate, seed=None):
#     """
#     Selects a percentage of participants from the array such that at least one is in the special list.

#     Args:
#         users_array (numpy.ndarray): Array of participants.
#         special_participant_list (list of lists): List of lists of special IDs for different activities.
#         training_rate (float): Percentage of participants to select (e.g., 0.3 for 30%).

#     Returns:
#         tuple: (selected_participants, remaining_participants)
#     """
#     set_seeds(seed)

#     if not 0 <= training_rate <= 1:
#         raise ValueError("Percentage must be between 0 and 1.")


#     special_sets = [set(special) for special in special_participant_list]

#     while True:
#         np.random.shuffle(users_array)
        
#         num_to_select = int(len(users_array) * training_rate)
#         train_participants = users_array[:num_to_select]
#         rest_participants = users_array[num_to_select:]
#         if all(special_set.intersection(train_participants) for special_set in special_sets):
#             print("Train participants are ", train_participants)       
#             print("Test participants are ", rest_participants)       
#             return train_participants, rest_participants
        
#         print("None of the special participants is included in training. Reshuffling...")


def select_participants(mode, case_study, users_array, special_participant_list, training_rate, seed=None):
    """
    Selects participants for training, validation, and testing, ensuring each group 
    has at least one participant from the special list.

    Args:
        users_array (numpy.ndarray): Array of participants.
        special_participant_list (list of lists): List of lists of special IDs for different activities.
        training_rate (float): Percentage of participants to select for training (e.g., 0.3 for 30%).
        seed (int, optional): Random seed for reproducibility.

    Returns:
        tuple: (train_participants, val_participants, test_participants)
    """
    set_seeds(seed)

    if not 0 <= training_rate <= 1:
        raise ValueError("Percentage must be between 0 and 1.")

    special_sets = [set(special) for special in special_participant_list]

    while True:
        np.random.shuffle(users_array)
        
        # Select training participants
        num_train = int(len(users_array) * training_rate)
        train_participants = users_array[:num_train]
        remaining_participants = users_array[num_train:]
        
        if 'base' in mode :
            if case_study == 'cv':

                # Split remaining participants into validation and testing
                mid_index = len(remaining_participants) // 2
                val_participants = remaining_participants[:mid_index]
                test_participants = remaining_participants[mid_index:]
                # Ensure each group contains at least one special participant
                if (all(special_set.intersection(train_participants) for special_set in special_sets) and
                    all(special_set.intersection(val_participants) for special_set in special_sets) and
                    all(special_set.intersection(test_participants) for special_set in special_sets)):
                    
                    # print("Train participants: ", train_participants)
                    # print("Validation participants: ", val_participants)
                    # print("Test participants: ", test_participants)
                    
                    return train_participants, val_participants, test_participants
                
                # print("Reshuffling as one or more groups lack special participants...")
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

def find_files(directory, search_string):
    matching_files = []  # List to store files that match the condition
    for root, dirs, files in os.walk(directory):  # Walk through all files in the directory
        for file in files:
            if search_string in file:  # Check if the search string is in the filename
                matching_files.append(os.path.join(root, file))  # Add full path to list
    return matching_files


def accumulate_participant_files(args, name, users_list):
    if f"data_{args.dataset_version}_{name}.npy" in os.listdir(os.path.join('dataset', args.dataset)):
        # print("Already here, loading file...", os.path.join('dataset', args.dataset, f"data_{args.dataset_version}_{name}.npy"))        
        data = np.load(os.path.join('dataset', args.dataset, f"data_{args.dataset_version}_{name}.npy")).astype(np.float32)
        labels = np.load(os.path.join('dataset', args.dataset, f"label_{args.dataset_version}_{name}.npy")).astype(np.float32)
        return data, labels
    else:
        data = []
        labels = []
        # Iterate through files in the input folder
        for participant in users_list:
            file_name = "P" + f"{participant:03}" + ".data"            
            # print(file_name)
            file_path = os.path.join('dataset', args.dataset, file_name)

            if not os.path.isfile(file_path):
                continue  # Skip directories or invalid files

            # Load participant file
            try:
                participant_data = np.load(file_path, allow_pickle=True)
                windows, activity_values, user_values = participant_data
                # print(type(windows[0]))
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
                continue

            printfirst = False
            # Process each window (dataframe) and corresponding labels
            for window, activity, user in zip(windows, activity_values, user_values):

                if printfirst:
                    print("window is ", window)
                    print("label is ", activity)
                    print("user is ", user)
                # Convert window (dataframe) to numpy array
                window_data = window.to_numpy(dtype=np.float32)

                # Ensure window_data length is a multiple of args.dataset_cfg.seq_len
                usable_length = (window_data.shape[0] // args.dataset_cfg.seq_len) * args.dataset_cfg.seq_len
                if usable_length == 0:
                    continue  # Skip windows too short for even one sequence
                window_data = window_data[:usable_length, :]

                # Reshape into sequences of (args.dataset_cfg.seq_len, features)
                reshaped_data = window_data.reshape(-1, args.dataset_cfg.seq_len, window_data.shape[1])

                # Create corresponding labels
                activity_label = np.full((reshaped_data.shape[0], args.dataset_cfg.seq_len, 1), activity, dtype=np.int32)
                user_label = np.full((reshaped_data.shape[0], args.dataset_cfg.seq_len, 1), user, dtype=np.int32)
                combined_label = np.concatenate((activity_label, user_label), axis=-1)  
                if printfirst:
                    print("window is ", window_data)
                    print("label is ", activity_label)
                    print("user is ", user_label)
                    printfirst = False
                # Append processed data and labels
                data.append(reshaped_data)
                labels.append(combined_label)

        # Concatenate all data and labels
        if data:
            data = np.concatenate(data, axis=0).astype(np.float32)
            labels = np.concatenate(labels, axis=0).astype(np.float32)

            # Save to .npy files for next use
            np.save(os.path.join('dataset', args.dataset, f"data_{args.dataset_version}_{name}.npy"), data)
            np.save(os.path.join('dataset', args.dataset, f"label_{args.dataset_version}_{name}.npy"), labels)
            
            print(f"Data and labels saved and returned. Data shape: {data.shape}, Label shape: {labels.shape}")
            return data, labels
        else:
            print("No data processed. Check input folder and file formats.")


def prepare_datasets_participants(args, training_rate=0.8, seed=None):

    """
    Arguments:
    - mode: d2d, cv, cross c24
    - training dataset
    - split rate
    - users for each dataset, including special participants
    
    I probably dont need to know the test dataset, only the mode

    if d2d:
        - take the number of participants defined by the split rate, ensuring that special participants are still inside
        - return train and validation participants from the same dataset
    elif cv:
        if users < 10:
            - apply loocv
            - sort the list, take the last as validation
            - shift the list 
            - exclude the special participant, if they exist, from loocv
            - return train, valid, test from the same dataset
        else:
            - take the split rate required. Again make sure that special participants are inside
            - return train, valid, test from the same dataset
    elif cross-c24:
        - make the split, make sure special participants are inside
        - return train, valid and the test users that have to be used in testing
    """

    if args.case_study == "cv":
        if args.dataset_cfg.user_ids == []:
            user_ids_array = np.array([i for i in range(1, args.dataset_cfg.user_label_size + 1)])
        else:
            user_ids_array = np.array(args.dataset_cfg.user_ids)
        train_users, valid_users, test_users = select_participants(args.mode, args.case_study, user_ids_array, args.dataset_cfg.required_user_ids, training_rate, seed)
        
        # train_users = np.array([12, 25, 1])
        # valid_users = np.array([47, 48, 2])

        data_train, labels_train = accumulate_participant_files(args, "train", train_users)
        data_val, labels_val = accumulate_participant_files(args, "val", valid_users)
        data_test, labels_test = accumulate_participant_files(args, "test", test_users)
        # print("training data shape", data_train.shape)
        # print("training label shape", labels_train.shape)
        # print("validation data shape", data_val.shape)
        # print("validation label shape", labels_val.shape)
        # print("testing data shape", data_test.shape)
        # print("testing label shape", labels_test.shape)


        unique_label_train, counts_train = np.unique(labels_train[:, :, :1], return_counts=True)
        unique_label_vali, counts_vali = np.unique(labels_val[:, :, :1], return_counts=True)
        unique_label_test, counts_test = np.unique(labels_test[:, :, :1], return_counts=True)
        # print('Train label distribution: ', dict(zip(unique_label_train, counts_train)))
        # print('Validation label distribution: ', dict(zip(unique_label_vali, counts_vali)))
        # print('Test label distribution: ', dict(zip(unique_label_test, counts_test)))

        labels_train = labels_train[:, 0, args.dataset_cfg.activity_label_index]
        labels_test = labels_test[:, 0, args.dataset_cfg.activity_label_index]
        labels_val = labels_val[:, 0, args.dataset_cfg.activity_label_index]
        return data_train, labels_train, data_val, labels_val, data_test, labels_test

    elif args.case_study == "d2d":
        raise NotImplementedError
    elif args.case_study == "cross24":
        raise NotImplementedError
    else:
        raise NotImplementedError


def balance_dataset(data, labels, ratio=2):
    unique_activity_labels = labels 
    activity_counts = Counter(unique_activity_labels)

    # print("Current activity label distribution (per sample):")
    # for activity, count in sorted(activity_counts.items()):
    #     print(f"Activity {activity}: {count}")

    min_class_count = min(activity_counts.values())
    max_allowed_count = min_class_count * ratio

    balanced_data = []
    balanced_labels = []

    for activity in activity_counts:
        activity_indices = np.where(unique_activity_labels == activity)[0]

        if len(activity_indices) > max_allowed_count:
            activity_indices = np.random.choice(activity_indices, max_allowed_count, replace=False)

        balanced_data.append(data[activity_indices])
        balanced_labels.append(labels[activity_indices].reshape(-1, 1)) 

    balanced_data = np.vstack(balanced_data)
    balanced_labels = np.vstack(balanced_labels)
    print(balanced_labels.shape)
    # balanced_counts = Counter(balanced_labels)  
    # for activity, count in sorted(balanced_counts.items()):
    #     print(f"Activity {activity}: {count}")

    return balanced_data, balanced_labels.reshape(-1)



def prepare_pretrain_dataset(data, labels, training_rate, seed=None):
    set_seeds(seed)
    data_train, label_train, data_vali, label_vali, data_test, label_test = partition_and_reshape(data, labels, label_index=0
                                                                                                  , training_rate=training_rate, vali_rate=0.1
                                                                                                  , change_shape=False)
    return data_train, label_train, data_vali, label_vali


def prepare_classifier_dataset(data, labels, label_index=0, training_rate=0.8, label_rate=1.0, change_shape=True
                               , merge=0, merge_mode='all', seed=None, balance=False):

    set_seeds(seed)
    data_train, label_train, data_vali, label_vali, data_test, label_test \
        = partition_and_reshape(data, labels, label_index=label_index, training_rate=training_rate, vali_rate=0.1
                                , change_shape=change_shape, merge=merge, merge_mode=merge_mode)
    set_seeds(seed)
    if balance:
        data_train_label, label_train_label, _, _ \
            = prepare_simple_dataset_balance(data_train, label_train, training_rate=label_rate)
    else:
        data_train_label, label_train_label, _, _ \
            = prepare_simple_dataset(data_train, label_train, training_rate=label_rate)
    return data_train_label, label_train_label, data_vali, label_vali, data_test, label_test


def partition_and_reshape(data, labels, label_index=0, training_rate=0.8, vali_rate=0.1, change_shape=True
                          , merge=0, merge_mode='all', shuffle=True):
    arr = np.arange(data.shape[0])
    if shuffle:
        np.random.shuffle(arr)
    data = data[arr]
    labels = labels[arr]
    train_num = int(data.shape[0] * training_rate)
    vali_num = int(data.shape[0] * vali_rate)
    data_train = data[:train_num, ...]
    data_vali = data[train_num:train_num+vali_num, ...]
    data_test = data[train_num+vali_num:, ...]
    t = np.min(labels[:, :, label_index])
    label_train = labels[:train_num, ..., label_index] - t
    label_vali = labels[train_num:train_num+vali_num, ..., label_index] - t
    label_test = labels[train_num+vali_num:, ..., label_index] - t
    if change_shape:
        data_train = reshape_data(data_train, merge)
        data_vali = reshape_data(data_vali, merge)
        data_test = reshape_data(data_test, merge)
        label_train = reshape_label(label_train, merge)
        label_vali = reshape_label(label_vali, merge)
        label_test = reshape_label(label_test, merge)
    if change_shape and merge != 0:
        data_train, label_train = merge_dataset(data_train, label_train, mode=merge_mode)
        data_test, label_test = merge_dataset(data_test, label_test, mode=merge_mode)
        data_vali, label_vali = merge_dataset(data_vali, label_vali, mode=merge_mode)
    # print('Train Size: %d, Vali Size: %d, Test Size: %d' % (label_train.shape[0], label_vali.shape[0], label_test.shape[0]))
    return data_train, label_train, data_vali, label_vali, data_test, label_test

def prepare_simple_dataset(data, labels, training_rate=0.2):
    arr = np.arange(data.shape[0])
    np.random.shuffle(arr)
    data = data[arr]
    labels = labels[arr]
    train_num = int(data.shape[0] * training_rate)
    data_train = data[:train_num, ...]
    data_test = data[train_num:, ...]
    t = np.min(labels)
    label_train = labels[:train_num] - t
    label_test = labels[train_num:] - t
    labels_unique = np.unique(labels)
    label_num = []
    for i in range(labels_unique.size):
        label_num.append(np.sum(labels == labels_unique[i]))
    # print('Label Size: %d, Unlabel Size: %d. Label Distribution: %s'
    #       % (label_train.shape[0], label_test.shape[0], ', '.join(str(e) for e in label_num)))
    return data_train, label_train, data_test, label_test


def prepare_simple_dataset_balance(data, labels, training_rate=0.8):
    labels_unique = np.unique(labels)
    label_num = []
    for i in range(labels_unique.size):
        label_num.append(np.sum(labels == labels_unique[i]))
    train_num = min(min(label_num), int(data.shape[0] * training_rate / len(label_num)))
    if train_num == min(label_num):
        print("Warning! You are using all of label %d." % label_num.index(train_num))
    index = np.zeros(data.shape[0], dtype=bool)
    for i in range(labels_unique.size):
        class_index = np.argwhere(labels == labels_unique[i])
        class_index = class_index.reshape(class_index.size)
        np.random.shuffle(class_index)
        temp = class_index[:train_num]
        index[temp] = True
    t = np.min(labels)
    data_train = data[index, ...]
    data_test = data[~index, ...]
    label_train = labels[index, ...] - t
    label_test = labels[~index, ...] - t
    print('Balance Label Size: %d, Unlabel Size: %d; Real Label Rate: %0.3f' % (label_train.shape[0], label_test.shape[0]
                                                               , label_train.shape[0] * 1.0 / labels.size))
    return data_train, label_train, data_test, label_test


def regularization_loss(model, lambda1, lambda2):
    l1_regularization = 0.0
    l2_regularization = 0.0
    for param in model.parameters():
        l1_regularization += torch.norm(param, 1)
        l2_regularization += torch.norm(param, 2)
    return lambda1 * l1_regularization, lambda2 * l2_regularization


def match_labels(labels, labels_targets):
    index = np.zeros(labels.size, dtype=np.bool)
    for i in range(labels_targets.size):
        index = index | (labels == labels_targets[i])
    return index


class Pipeline():
    """ Pre-process Pipeline Class : callable """
    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError


class Preprocess4Normalization(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, feature_len, norm_acc=True, norm_mag=True, gamma=1.0):
        super().__init__()
        self.feature_len = feature_len
        self.norm_acc = norm_acc
        self.norm_mag = norm_mag
        self.eps = 1e-5
        self.acc_norm = 9.8
        self.gamma = gamma

    def __call__(self, instance):
        instance_new = instance.copy()[:, :self.feature_len]
        if instance_new.shape[1] == 3 and self.norm_acc:
            instance_new[:, :3] = instance_new[:, :3] / self.acc_norm
        if instance_new.shape[1] >= 6 and self.norm_acc:
            instance_new[:, :3] = instance_new[:, :3] / self.acc_norm
        if instance_new.shape[1] == 9 and self.norm_mag:
            mag_norms = np.linalg.norm(instance_new[:, 6:9], axis=1) + self.eps
            mag_norms = np.repeat(mag_norms.reshape(mag_norms.size, 1), 3, axis=1)
            instance_new[:, 6:9] = instance_new[:, 6:9] / mag_norms * self.gamma
        return instance_new


class Preprocess4Mask:
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, mask_cfg):
        self.mask_ratio = mask_cfg.mask_ratio  # masking probability
        self.mask_alpha = mask_cfg.mask_alpha
        self.max_gram = mask_cfg.max_gram
        self.mask_prob = mask_cfg.mask_prob
        self.replace_prob = mask_cfg.replace_prob

    def gather(self, data, position1, position2):
        result = []
        for i in range(position1.shape[0]):
            result.append(data[position1[i], position2[i]])
        return np.array(result)

    def mask(self, data, position1, position2):
        for i in range(position1.shape[0]):
            data[position1[i], position2[i]] = np.zeros(position2[i].size)
        return data

    def replace(self, data, position1, position2):
        for i in range(position1.shape[0]):
            data[position1[i], position2[i]] = np.random.random(position2[i].size)
        return data

    def __call__(self, instance):
        shape = instance.shape

        # the number of prediction is sometimes less than max_pred when sequence is short
        n_pred = max(1, int(round(shape[0] * self.mask_ratio)))

        # For masked Language Models
        # mask_pos = bert_mask(shape[0], n_pred)
        mask_pos = span_mask(shape[0], self.max_gram,  goal_num_predict=n_pred)

        instance_mask = instance.copy()

        if isinstance(mask_pos, tuple):
            mask_pos_index = mask_pos[0]
            if np.random.rand() < self.mask_prob:
                self.mask(instance_mask, mask_pos[0], mask_pos[1])
            elif np.random.rand() < self.replace_prob:
                self.replace(instance_mask, mask_pos[0], mask_pos[1])
        else:
            mask_pos_index = mask_pos
            if np.random.rand() < self.mask_prob:
                instance_mask[mask_pos, :] = np.zeros((len(mask_pos), shape[1]))
            elif np.random.rand() < self.replace_prob:
                instance_mask[mask_pos, :] = np.random.random((len(mask_pos), shape[1]))
        seq = instance[mask_pos_index, :]
        return instance_mask, np.array(mask_pos_index), np.array(seq)


class IMUDataset(Dataset):
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(self, data, labels, pipeline=[]):
        super().__init__()
        self.pipeline = pipeline
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        instance = self.data[index]
        for proc in self.pipeline:
            instance = proc(instance)
        return torch.from_numpy(instance).float(), torch.from_numpy(np.array(self.labels[index])).long()

    def __len__(self):
        return len(self.data)


class FFTDataset(Dataset):
    def __init__(self, data, labels, mode=0, pipeline=[]):
        super().__init__()
        self.pipeline = pipeline
        self.data = data
        self.labels = labels
        self.mode = mode

    def __getitem__(self, index):
        instance = self.data[index]
        for proc in self.pipeline:
            instance = proc(instance)
        seq = self.preprocess(instance)
        return torch.from_numpy(seq), torch.from_numpy(np.array(self.labels[index])).long()

    def __len__(self):
        return len(self.data)

    def preprocess(self, instance):
        f = np.fft.fft(instance, axis=0, n=10)
        mag = np.abs(f)
        phase = np.angle(f)
        return np.concatenate([mag, phase], axis=0).astype(np.float32)


class LIBERTDataset4Pretrain(Dataset):
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(self, data, pipeline=[]):
        super().__init__()
        self.pipeline = pipeline
        self.data = data

    def __getitem__(self, index):
        instance = self.data[index]
        for proc in self.pipeline:
            instance = proc(instance)
        mask_seq, masked_pos, seq = instance
        return torch.from_numpy(mask_seq), torch.from_numpy(masked_pos).long(), torch.from_numpy(seq)

    def __len__(self):
        return len(self.data)


def handle_argv(target, config_train, prefix):
    parser = argparse.ArgumentParser(description='PyTorch LIMU-BERT Model')
    parser.add_argument('case_study', type=str, help='The type of study I am running', choices=['cv', 'd2d', 'cross24'])
    parser.add_argument('model_version', type=str, help='Model config')
    parser.add_argument('dataset', type=str, help='Dataset name', choices=['hhar', 'motion', 'uci', 'shoaib', 'c24'])
    parser.add_argument('dataset_version',  type=str, help='Dataset version', choices=['10_100', '20_120', '25_125'])
    parser.add_argument('-g', '--gpu', type=str, default=None, help='Set specific GPU')
    parser.add_argument('-f', '--model_file', type=str, default=None, help='Pretrain model file')
    parser.add_argument('-t', '--train_cfg', type=str, default='./config/' + config_train, help='Training config json file path')
    parser.add_argument('-a', '--mask_cfg', type=str, default='./config/mask.json',
                        help='Mask strategy json file path')
    parser.add_argument('-l', '--label_index', type=int, default=-1,
                        help='Label Index')
    parser.add_argument('-s', '--save_model', type=str, default='model',
                        help='The saved model name')
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    model_cfg = load_model_config(target, prefix, args.model_version)
    if model_cfg is None:
        print("Unable to find corresponding model config!")
        sys.exit()
    args.model_cfg = model_cfg
    dataset_cfg = load_dataset_stats(args.dataset, args.dataset_version)
    if dataset_cfg is None:
        print("Unable to find corresponding dataset config!")
        sys.exit()
    args.dataset_cfg = dataset_cfg
    args = create_io_config(args, args.dataset, args.dataset_version, pretrain_model=args.model_file, target=target)
    args.mode = prefix
    return args



def handle_argv_simple():
    parser = argparse.ArgumentParser(description='PyTorch LIMU-BERT Model')
    parser.add_argument('model_file', type=str, default=None, help='Pretrain model file')
    parser.add_argument('dataset', type=str, help='Dataset name', choices=['hhar', 'motion', 'uci', 'shoaib', 'c24','merge'])
    parser.add_argument('dataset_version',  type=str, help='Dataset version', choices=['10_100', '20_120', '25_125'])
    args = parser.parse_args()
    dataset_cfg = load_dataset_stats(args.dataset, args.dataset_version)
    if dataset_cfg is None:
        print("Unable to find corresponding dataset config!")
        sys.exit()
    args.dataset_cfg = dataset_cfg
    return args


def load_raw_data(args):
    data = np.load(args.data_path).astype(np.float32)
    labels = np.load(args.label_path).astype(np.float32)
    return data, labels


def load_pretrain_data_config(args):
    model_cfg = args.model_cfg
    train_cfg = TrainConfig.from_json(args.train_cfg)
    mask_cfg = MaskConfig.from_json(args.mask_cfg)
    dataset_cfg = args.dataset_cfg
    if model_cfg.feature_num > dataset_cfg.dimension:
        print("Bad Crossnum in model cfg")
        sys.exit()
    set_seeds(train_cfg.seed)
    data = np.load(args.data_path).astype(np.float32)
    # data = data[:,:,:3]
    labels = np.load(args.label_path).astype(np.float32)
    return data, labels, train_cfg, model_cfg, mask_cfg, dataset_cfg

def load_pretrain_config(args):
    model_cfg = args.model_cfg
    train_cfg = TrainConfig.from_json(args.train_cfg)
    mask_cfg = MaskConfig.from_json(args.mask_cfg)
    dataset_cfg = args.dataset_cfg
    if model_cfg.feature_num > dataset_cfg.dimension:
        print("Bad Crossnum in model cfg")
        sys.exit()
    set_seeds(train_cfg.seed)
    return train_cfg, model_cfg, mask_cfg, dataset_cfg


def load_classifier_data_config(args):
    model_cfg = args.model_cfg
    train_cfg = TrainConfig.from_json(args.train_cfg)
    dataset_cfg = args.dataset_cfg
    set_seeds(train_cfg.seed)
    data = np.load(args.data_path).astype(np.float32)
    labels = np.load(args.label_path).astype(np.float32)
    return data, labels, train_cfg, model_cfg, dataset_cfg


def load_classifier_config(args):
    model_cfg = args.model_cfg
    train_cfg = TrainConfig.from_json(args.train_cfg)
    dataset_cfg = args.dataset_cfg
    set_seeds(train_cfg.seed)
    return train_cfg, model_cfg, dataset_cfg


def load_bert_classifier_data_config(args):
    model_bert_cfg, model_classifier_cfg = args.model_cfg
    print(model_bert_cfg)
    train_cfg = TrainConfig.from_json(args.train_cfg)
    dataset_cfg = args.dataset_cfg
    if model_bert_cfg.feature_num > dataset_cfg.dimension:
        print("Bad feature_num in model cfg")
        sys.exit()
    set_seeds(train_cfg.seed)
    data = np.load(args.data_path).astype(np.float32)
    labels = np.load(args.label_path).astype(np.float32)
    return data, labels, train_cfg, model_bert_cfg, model_classifier_cfg, dataset_cfg

def load_bert_classifier_config(args):
    model_bert_cfg, model_classifier_cfg = args.model_cfg
    print(model_bert_cfg)
    train_cfg = TrainConfig.from_json(args.train_cfg)
    dataset_cfg = args.dataset_cfg
    if model_bert_cfg.feature_num > dataset_cfg.dimension:
        print("Bad feature_num in model cfg")
        sys.exit()
    set_seeds(train_cfg.seed)
    return train_cfg, model_bert_cfg, model_classifier_cfg, dataset_cfg

def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
