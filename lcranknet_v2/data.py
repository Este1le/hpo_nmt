import torch
import torch.nn.functional as F
from torch.utils.data import Sampler
from collections import defaultdict
import pickle, os 
import math
from copy import deepcopy
import itertools
from torch.utils.data import Subset

from utils import *

class LCDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_cfg, split, logger):
        super(LCDataset, self).__init__()
        self.split = split
        self.dataset_cfg = dataset_cfg
        self.test_dataset = dataset_cfg['test']
        self.num_test = dataset_cfg['num_test']
        self.logger = logger
        self.load_data()
        self.num_curves_per_dataset = self.get_num_curves_per_dataset()
        self.dev_data = []
        self.get_train_data()
        if self.split == "dev":
            self.get_dev_data()
        
    
    def get_ckpt_evaled(self, data_item):
        # Get the checkpoint ids when a termination decision has to be made in SH
        max_len = data_item['max_len']
        num_models = data_item['num_models']
        percentage = 0.75
        num_cuts = int(math.log(num_models, 2))
        cut_freq = int(max_len*percentage/num_cuts)
        ckpt_evaled = [(i*cut_freq)-1 for i in range(1, num_cuts+1)]
        return ckpt_evaled
    
    def load_data(self):
        with open(self.dataset_cfg['data_file'], 'rb') as f:
            data_pkl = pickle.load(f)
        self.data = []
        for data_item in data_pkl:
            data_dict = {}
            data_dict['task'] = TASKS.index(data_item['task'])
            data_dict['dataset_name'] = DATASETS.index(data_item['dataset_name'])
            data_dict['src'] = SRCS.index(data_item['src'])
            data_dict['trg'] = TRGS.index(data_item['trg'])
            data_dict['basemodel'] = BASEMODELS.index(data_item['basemodel'])
            data_dict['hyperparameters']  = torch.tensor(list(data_item['hyperparams'].values())) # 8
            data_dict['optimal'] = torch.tensor(data_item['optimal'])
            data_dict['num_models'] = data_item['num_models']
            data_dict['max_len'] = data_item['max_len']
            ckpt_evaled = self.get_ckpt_evaled(data_item)
            for cev in ckpt_evaled:
                if len(data_item['curve']) > cev:
                    partial_curve = [ckpt for ckpt in data_item['curve'][:cev+1]]
                    valid_partial_curve = []
                    for i in range(len(partial_curve)):
                        val = partial_curve[i]
                        if val > 10e2:
                            valid_partial_curve.append(partial_curve[i-1])
                        else:
                            valid_partial_curve.append(val)
                    data_dict['curve'] = torch.tensor(valid_partial_curve) 
                    # remove values that are too large
                    self.data.append(deepcopy(data_dict))
    
    def get_num_curves_per_dataset(self):
        num_curves_per_dataset = defaultdict(int)
        for data_dict in self.data:
            dataset = data_dict['dataset_name']
            num_curves_per_dataset[dataset] += 1
        return num_curves_per_dataset
    
    def get_train_data(self):
        new_data = []
        dataset_num_data = defaultdict(int)
        for data_dict in self.data:
            dataset = data_dict['dataset_name']
            if dataset == DATASETS.index(self.test_dataset):
                if dataset_num_data[dataset] > self.num_test:
                    self.dev_data.append(data_dict)
                    continue
            new_data.append(data_dict)
            dataset_num_data[dataset] += 1
        self.logger.info("#Train curves: " + str(len(new_data)))
        self.data = new_data
    
    def get_dev_data(self):
        self.logger.info("#Dev curves: " + str(len(self.dev_data)))
        self.data = self.dev_data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def build_dataset(dataset_cfg, split, logger):
    dataset = LCDataset(dataset_cfg, split, logger)
    return dataset

def pad_to_max(curve, cfg):
    pad_length = max(0, cfg['data']['max_length'] - curve.size(0) + 1)
    padded = F.pad(curve, (0, pad_length), "constant", cfg['data']['pad_id'])
    #padded = F.pad(curve, (0, pad_length), "constant", curve[-1].item())
    return padded

def collate_fn_(inputs, cfg):
    outputs = {
        "hyperparameters": [i['hyperparameters'] for i in inputs],
        "optimals": [i['optimal'] for i in inputs],
        "curves": [pad_to_max(i['curve'], cfg) for i in inputs],
        "datasets": [i['dataset_name'] for i in inputs],
        "tasks": [i['task'] for i in inputs],
        "srcs": [i['src'] for i in inputs],
        "trgs": [i['trg'] for i in inputs],
        "basemodels": [i['basemodel'] for i in inputs]
    }
    return outputs

class LengthBatchSampler(Sampler):
    # Sampler: yields a list of batch indices at a time can be passed as the batch_sampler argument
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size if batch_size%2==0 else 2*batch_size
        self.length_indices = self._group_indices_by_lengths()

    def _group_indices_by_lengths(self):
        length_indices = {}
        for idx, item in enumerate(self.dataset):
            if len(item['curve']) not in length_indices:
                length_indices[len(item['curve'])] = []
            length_indices[len(item['curve'])].append(idx)
        for length, lis in length_indices.items():
            pairs = list(itertools.combinations(lis, 2))
            length_indices[length] = pairs
        return length_indices
    
    def __iter__(self):
        # if self.split == "dev":
        #     subset_size = 50
        # else:
        #     subset_size = -1
        batch = []
        for pairs in self.length_indices.values():
            #random.shuffle(pairs)
            # pairs = pairs[:subset_size]
            for pair in pairs:
                batch += pair
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
                    
    def __len__(self):
        batch_size = self.batch_size if self.batch_size%2==0 else 2*self.batch_size
        return int(sum([len(pairs) for pairs in self.length_indices.values()]) // (batch_size//2))

def build_dataloader(cfg, dataset, split="train"):
    sampler = LengthBatchSampler(dataset, cfg['training']['batch_size'])
    dataloader = torch.utils.data.DataLoader(dataset, 
                                            collate_fn=lambda x: collate_fn_(
                                                inputs=x,
                                                cfg=cfg),
                                            num_workers=cfg['training']['num_workers'],
                                            batch_sampler=sampler)
    return dataloader, sampler