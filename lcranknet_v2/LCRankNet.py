import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.nn.init as init
from utils import *

class LCRankNet(nn.Module):
    def __init__(self, cfg):
        super(LCRankNet, self).__init__()
        self.device = cfg['device']
        embed_dim = int(cfg['training']['embed_dim'])
        num_hyperparams = int(cfg['training']['num_hyperparams'])
        curve_output_size = int(cfg['training']['curve_output_size'])
        ff_size = int(cfg['training']['ff_size'])
        dropout_prob = float(cfg['training']['dropout'])
        
        # Convolutional layers for processing partial learning curve
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout_prob)
        for k in range(2, 6):
            conv = nn.Conv1d(in_channels=1, out_channels=curve_output_size, kernel_size=k)
            #init.xavier_uniform_(conv.weight)
            init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='leaky_relu')
            if conv.bias is not None:
                init.zeros_(conv.bias)
            bn = nn.BatchNorm1d(num_features=curve_output_size)
            self.conv_layers.append(conv)
            self.bn_layers.append(bn)

        # Embedding
        num_datasets = len(DATASETS)
        self.dataset_embedding = nn.Embedding(num_datasets, embed_dim)
        num_tasks = len(TASKS)
        self.task_embedding = nn.Embedding(num_tasks, embed_dim)
        num_srcs = len(SRCS)
        self.src_embedding = nn.Embedding(num_srcs, embed_dim)
        num_trgs = len(TRGS)
        self.trg_embedding = nn.Embedding(num_trgs, embed_dim)
        num_basemodels = len(BASEMODELS)
        self.basemodel_embedding = nn.Embedding(num_basemodels, embed_dim)

        # Fully connected layer for the final prediction
        self.fc1 = nn.Linear(len(self.conv_layers) * curve_output_size + \
            embed_dim * 5 + num_hyperparams, ff_size)
        #init.xavier_uniform_(self.fc1.weight)
        init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='leaky_relu')
        if self.fc1.bias is not None:
            init.zeros_(self.fc1.bias)
        self.fc2 = nn.Linear(ff_size, 1)
        #init.xavier_uniform_(self.fc2.weight)
        init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='leaky_relu')
        if self.fc2.bias is not None:
            init.zeros_(self.fc2.bias)
        self.relu = nn.LeakyReLU(0.1)
        self.elu = nn.ELU()
        
    def forward(self, curves, datasets, tasks, srcs, trgs, basemodels, hyperparameters, optimals):
        datasets = torch.tensor(datasets, dtype=torch.int32).to(self.device)
        dataset_embed = self.dataset_embedding(datasets.view(-1,1))
        tasks = torch.tensor(tasks, dtype=torch.int32).to(self.device)
        task_embed = self.task_embedding(tasks.view(-1, 1))
        srcs = torch.tensor(srcs, dtype=torch.int32).to(self.device)
        src_embed = self.src_embedding(srcs.view(-1, 1))
        trgs = torch.tensor(trgs, dtype=torch.int32).to(self.device)
        trg_embed = self.trg_embedding(trgs.view(-1, 1))
        basemodels = torch.tensor(basemodels, dtype=torch.int32).to(self.device)
        basemodel_embed = self.basemodel_embedding(basemodels.view(-1, 1))

        # Applying convolutions and max pooling to the learning curve input
        conv_results = []
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            conv_out = conv(torch.stack(curves).unsqueeze(1).to(self.device))
            bn_out = self.relu(bn(conv_out))
            dropout_out = self.dropout(bn_out)
            pooled_out = F.max_pool1d(dropout_out, dropout_out.size(2)).squeeze(2)
            conv_results.append(pooled_out)
        # Concatenating all features
        concatenated_features = torch.cat([*conv_results, dataset_embed.squeeze(1), 
            task_embed.squeeze(1), src_embed.squeeze(1), trg_embed.squeeze(1),
            basemodel_embed.squeeze(1),
            torch.stack(hyperparameters).to(self.device)], 1)
        # Final fully connected layer
        intermediate = self.relu(self.fc1(concatenated_features))
        mu = self.fc2(intermediate)
        sigma = self.elu(self.fc2(intermediate)).squeeze(-1) + 1
        return mu, sigma

def build_model(cfg):
    model = LCRankNet(cfg)
    return model.to(cfg['device'])