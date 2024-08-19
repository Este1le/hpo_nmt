import torch
import yaml
import os
import numpy as np
import random

# https://github.com/Este1le/hpo_nmt
DATASETS = ['material-so-en', 'material-sw-en', 'robust19-ja-en', 'robust19-en-ja',
            'ted-ru-en', 'ted-zh-en', 'zh-en', 'de-en', 'fr-en']
TASKS = ['scratch', 'finetune']
SRCS = ['so', 'sw', 'ja', 'en', 'ru', 'zh', 'de', 'fr']
TRGS = ['en', 'ja']
BASEMODELS = ['bloomz', 'xglm', 'scratch']

def set_seed(seed: int):
    """
    Set the random seed for modules torch, numpy and random.
    :param seed: random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_config(path="configs.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.
    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg
