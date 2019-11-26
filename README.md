# Datasets for Hyperparameter Optimization of Neural Machine Translation

We collected a large amount of trained NMT models (Transformers) covering a wide range of hyperparameters and record their hyperparameter configurations and performance measurements, in order to speed up HPO experiments. When evaluating a HPO method, a developer can look up the model performance whenever necessary, without having to train a NMT model from scratch. Specifically, we trained NMT models on six different parallel corpora: zh-en, ru-en, ja-en, en-ja, sw-en, so-en. 

### HPO Datasets
`cd datasets`

1. `*.hyps`: Hyperparameter configurations:
	
	bpe\_symbols, num\_layers, num\_embed, transformer\_feed\_forward\_num\_hidden, transformer\_attention\_heads, initial\_learning\_rate
	
2. `*.evals`: Performance measurements:

	dev\_bleu, dev\_gpu\_time, dev\_ppl, num\_updates, gpu\_memory, num\_param
	
	*If working with gpu\_memory, please filter out the models with 0 gpu\_memory.*

3. `*.fronts`: Pareto-optimal points for (dev\_bleu, dev\_gpu\_time). `1` indicates it is a Pareto-optimal point. `0` indicates it is not a Pareto-optimal point.

### Evaluation Scripts
`cd scripts`
 
1. Evaluate HPO methods on single-objective optimization.
	`python ./eval_single.py -s ./scripts/examples/example.ss -e ./scripts/examples/example.bleu -i 3`

2. Evaluate HPO methods on multi-objective optimization.
	`python ./scripts/eval_multiple.py -s ./scripts/examples/example.ss -f ./scripts/examples/example.fronts -i 3`



