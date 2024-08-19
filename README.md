# Benchmarks for Hyperparameter Optimization of Neural Machine Translation Systems

We pre-train a large set of neural machine translation (NMT) systems (Transformers) and record their configurations, learning curves, and evaluation results in a table. This allows for efficient evaluation of Hyperparameter Optimization (HPO) by looking up the table as needed, without training each model from scratch, significantly speeding up the experimental process. We release two benchmark datasets: `NMTHPO` and `NMTLC` that serve different research purposes.


# I. NMTHPO Dataset
This dataset ([Zhang and Duh, 2020](https://aclanthology.org/2020.tacl-1.26/) )includes 2,245 trained-from-scratch NMT systems trained on 6 different corpora. A variety of performance metrics are reported for each hyperparameter configuration. This dataset is well-suited for evaluating black-box HPO methods such as random search or Bayesian optimization. It can also be utilized to assess multi-objective optimization algorithms.

`cd datasets`

1. `*.hyps`: Hyperparameter configurations:
	
	bpe\_symbols, num\_layers, num\_embed, transformer\_feed\_forward\_num\_hidden, transformer\_attention\_heads, initial\_learning\_rate
	
2.  `*.hyps_scaled`: Scaled hyperparametr configurations within range [0,1].

	|language      | hyperparameter | domain        | scaling function |
	|--------------| ---------------| --------------| -----------------|
	|zh, ru, ja, en| bpe            | 10, 30, 50 (k) | (x-10k)/40k |
	| sw,so        | bpe            | 1, 2, 4, 8, 16, 32 (k) | log(x/1000)/5 | 
	|zh, ru, ja, en | num_layers     | 2,4         |   (x-2)/2 |
	|sw | num_layers     | 1, 2,4,6  |   (x-1)/5 |
	|so | num_layers     | 1, 2,4    |   (x-1)/3 |
	|zh, ru, ja, en, sw, so | num_embed      | 256, 512, 1024 | (log(x)-8)/2 |
	|zh, ru, ja, en, sw, so | num_hidden     | 1024, 2048 |  (x-1024)/1024 | 
	|zh, ru, ja, en, sw, so | num_heads      | 8, 16 | (x-8)/8 |
	|zh, ru, ja, en, sw, so | learning_rate  | 3, 6, 10 (10-4) | (x-0.0003)/0.0007 | 
	
	
3. `*.evals`: Performance measurements:

	dev\_bleu, dev\_gpu\_time, dev\_ppl, num\_updates, gpu\_memory, num\_param
	
	*If working with gpu\_memory, please filter out the models with 0 gpu\_memory.*
	
4. `*.fronts`: Pareto-optimal points for (dev\_bleu, dev\_gpu\_time). `1` indicates it is a Pareto-optimal point. `0` indicates it is not a Pareto-optimal point.


### Evaluation Scripts
`cd scripts`
 
1. Evaluate HPO methods on single-objective optimization.
	`python ./eval_single.py -s ./scripts/examples/example.ss -e ./scripts/examples/example.bleu -i 3`

2. Evaluate HPO methods on multi-objective optimization.
	`python ./scripts/eval_multiple.py -s ./scripts/examples/example.ss -f ./scripts/examples/example.fronts -i 3`

# II. NMTLC Dataset

This dataset ([Zhang and Duh, 2024](https://www.cs.jhu.edu/~xzhan138/papers/AMTA2024_LC.pdf)) includes 2,469 models trained on 9 different corpora, extending the `NMTHPO` dataset by NMT systems fine-tuned from LLMs. This is the first HPO benchmark dataset to include models fine-tuned from LLMs for NMT tasks. We report the learning curves (perplexities and BLEU scores on the development set at each checkpoint during training) and the optimal performance for each hyperparameter configuration. It can be used for evaluating multi-fidelity / gray-box HPO methods such as successive halving, and single-objective optimization algorithms. 

`NMTLC` contains models trained on 9 corpora:

| task | language pair | #configuration | 
| --- | --- | ---|
| trained_from_scratch | zh-en | 118 |
| trained_from_scratch | ru-en | 176 |
| trained_from_scratch | ja-en | 150 |
| trained_from_scratch | en-ja | 168 |
| trained_from_scratch | sw-en | 767 |
|  trained_from_scratch | so-en | 604 |
| fine-tuned | zh-en | 162 |
|fine-tuned | de-en | 162 |
|fine-tuned|fr-en|162| 




The hyperparameter search space for the trained-from-scratch systems are the same as in `NMTHPO`. For NMT models fine-tuned from LLMs, we consider four hyperparameters to define the search space:

- **LLM**: BLOOMZ 560m, 1b7, and 3b, XGLM 564M, 1.7B, and 2.9B.
- **LoRA rank**: 2, 16, and 64.
- **Batch size**: 16, 32, and 64.
- **Learning rate**: 2e-5, 1e-4, and 2e-4

`lc_datasets/nmtlc.pkl` contains all the samples of the `NMTLC` dataset. 

A sample of a trained-from-scratch NMT system in `nmtlc.pkl`:
```
{'task': 'scratch', 

'dataset_name': 'material-so-en', 

'src': 'so', 'trg': 'en', 

'basemodel': 'scratch', 

# The LoRA rank r is set to 0 for trained-from-scratch models.
'hyperparams': {'dataset_size': 24000, 'initial_lr': 0.0003, 'model_size': 1024, 'ff_num_hidden': 1024, 'num_layers': 1, 'batch_size': 4096, 'bpe': 1, 'r': 0}, 

# The perplexities on the development set at each checkpoint
'perplexity_curve': [18.195837, 21.232226, 25.462998, 27.990129, 29.509712, 30.254892, 30.075628, 30.241507, 30.497135, 21.220965, 25.037733, 28.287772, 29.335146, 30.260893, 30.640363, 30.687425, 30.752129],

# The lowest perplexity obtained on the development set
'perplexity_optimal': 18.195837, 

# Number of checkpoints
'max_len': 77, 
}
```

A sample of a fine-tuned system in `nmtlc.pkl`:
```
{'task': 'finetune', 
'dataset_name': 'fr-en', 
'src': 'fr', 'trg': 'en', 
'basemodel': 'xglm', 
'hyperparams': {'dataset_size': 403857, 'initial_lr': 0.0002, 'model_size': 2048, 'ff_num_hidden': 8192, 'num_layers': 48, 'batch_size': 64, 'bpe': 256008, 'r': 64}, 
'perplexity_curve': [3.539940595626831, 3.2750587463378906, 2.8639228343963623, 2.332699775695801, 2.2487266063690186, 2.2674310207366943, 2.269585132598877, 2.2674617767333984, 2.243283987045288, 2.2739784717559814, 2.2900540828704834, 2.301405906677246, 2.321772813796997, 2.337951898574829, 2.3834662437438965, 2.3704450130462646, 2.385355234146118, 2.3497314453125, 2.3883020877838135, 2.417386293411255], 
'perplexity_optimal': 2.243283987045288, 
'bleu_curve': [8.98, 15.46, 17.06, 12.67, 15.73, 20.36, 21.99, 21.63, 21.9, 23.62, 23.31, 22.59, 20.08, 17.3, 19.18, 20.48, 16.8, 22.73, 18.72, 21.27], 
'bleu_optimal': 23.62,
'max_len': 74}
```
### Statistics
BLEU distribution on the hyperparameter search space of the `NMTLC` benchmark dataset:
<img src="images/bleu_dist.png" alt="BLEU distribution" title="BLEU distrbution on the hyperparameter search space of the NMTLC dataset." width="550" height="360">

Learning curve length distribution on the hyperparameter search space of the `NMTLC` dataset:
<img src="images/length_dist.png" alt="Learning curve length distribution" title="Learning curve length distrbution on the hyperparameter search space of the NMTLC dataset." width="550" height="360">

# Citation
```
@inproceedings{zhang2024best,
  	title={Best Practices of Successive Halving on Neural Machine Translation and Large Language Models},
  	author={Zhang, Xuan and Duh, Kevin},
  	booktitle={Proceedings of the 16th biennial conference of the Association for Machine Translation in the Americas (Volume 1: Research Track)},
  	year={2024}
}

@article{zhang-duh-nmthpo20,
	author={Zhang, Xuan and Duh, Kevin},
	title={Reproducible and Efficient Benchmarks for Hyperparameter Optimization of Neural Machine Translation Systems},
	booktitle={Transactions of the Association for Computational Linguistics},
	year={2020}
}
```

