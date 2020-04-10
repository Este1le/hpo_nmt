# Reproducible and Efficient Benchmarks for Hyperparameter Optimization of Neural Machine Translation Systems

We collected a large amount of trained NMT models (Transformers) covering a wide range of hyperparameters and record their hyperparameter configurations and performance measurements, in order to speed up HPO experiments. When evaluating a HPO method, a developer can look up the model performance whenever necessary, without having to train a NMT model from scratch. Specifically, we trained NMT models on six different parallel corpora: zh-en, ru-en, ja-en, en-ja, sw-en, so-en. 

### HPO Datasets
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

### Citation
```
@InProceedings{zhang-duh-nmthpo20,
	       author={Zhang, Xuan and Duh, Kevin},
	       title={Reproducible and Efficient Benchmarks for Hyperparameter Optimization of Neural Machine Translation Systems},
	       booktitle={Transactions of the Association for Computational Linguistics},
	       year={2020}
}
```

