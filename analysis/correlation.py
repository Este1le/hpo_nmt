#!/usr/bin/env python3

import numpy as np
import os
import scipy.stats

datasetdir='../datasets'
tasks=['zh-en', 'ru-en', 'ja-en','en-ja', 'so-en', 'sw-en']
#tasks=['ja-en', 'so-en']

# dict that indexes into matrix output of read_dataset()
c=dict(zip(['bpe_symbols', 'num_layers', 'num_embed', 
            'transformer_feed_forward_num_hidden',
            'transformer_attention_heads', 'initial_learning_rate',
            'dev_bleu', 'dev_gpu_time', 'dev_ppl', 'num_updates', 
            'gpu_memory', 'num_param','num_param_nonembed'],
            range(13)))


def read_dataset(datasetdir, task):
    # reading files
    hyps_file=os.path.join(datasetdir, f"{task}.hyps")
    evals_file=os.path.join(datasetdir, f"{task}.evals")
    hyps = np.genfromtxt(hyps_file, delimiter='\t')
    evals = np.genfromtxt(evals_file, delimiter='\t')

    # estimate model size without input/output embeddings
    # subtract source side (BPE*embed) & target side (2*BPE*embed + BPE)
    num_param_nonembed=np.array([evals[:,-1] - 
                                (3*hyps[:,0]*hyps[:,2]+hyps[:,0])]).T

    # return all in a matrix (each row = sample, columns indexed by dict "c")
    return np.hstack((hyps, evals, num_param_nonembed))


def spearman_correlation_matrix(d):
    corr = np.full((d.shape[1],d.shape[1]), -1.0, dtype=float)
    for i in range(d.shape[1]):
        for j in range(d.shape[1]):
            corr[i,j], _ = scipy.stats.spearmanr(d[:,i],d[:,j])
    return corr


def print_correlation_matrix(corr):
    row_format = "{:6.2f} " * corr.shape[1]
    row_format2 = "{:6} " * corr.shape[1]
    print(row_format2.format(*range(corr.shape[1])))
    print('-------'*corr.shape[1])
    for metric in ['dev_bleu', 'dev_ppl', 'dev_gpu_time']:
        print(row_format.format(*corr[c[metric]]), metric)        

# print results
print('\n'.join(["  %d: %s"%(v,k) for k,v in c.items()]))
for task in tasks:
    d = read_dataset(datasetdir, task)

    print(f"\nPearson correlation for task: {task}")
    corr_pearson = np.corrcoef(d,rowvar=False)
    print_correlation_matrix(corr_pearson)

    print(f"\nSpearman correlation for task: {task}")
    corr_spearman = spearman_correlation_matrix(d)
    print_correlation_matrix(corr_spearman)

#np.set_printoptions(precision=2, threshold=np.inf, suppress=True)
#print(corr_spearman)
