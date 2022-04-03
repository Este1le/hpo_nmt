This is the starting kit for the AutoML2022 competition

## Quick walkthrough

(1) To run a hyperparameter optimization (HPO) and generate predictions, run the following: 
```
# python ingestion_program/ingestion.py $inputdir $predictiondir ./ingestion_program $codedir 
# e.g.
python ingestion_program/ingestion.py ../../datasets ./pred ./ingestion_program ./sample_code_submission
```

Here, `$inputdir` refers to the directory that contains the data for hyperparameter optimization under the tabular benchmark framework.
For example, in the github `https://github.com/Este1le/hpo_nmt/tree/master/datasets` (also linked by `../../datasets`), the file `so-en.hyps` represent the hyperparameter configurations of various Transformer architectures for the "so-en" MT dataset, and the file `so-en.evals` represent the associated speed/accuracy metrics for each row in `so-en.hyps`. 

Please set `$predictiondir` (also called output_dir in codalab ingestion) to the directory where your model predictions will be stored.
Further, set `$codedir` to point to the directory of your HPO code. 
Note the sample_code_submission directory includes the file `hpo_model.py`. Currently its set as a random search baseline.
To implement your own HPO method, please modify the run() function. 

Specifically, the ingestion program will read run hpo_model.run() using different benchmark datasets.
Each time, it will generate a sample sequence prediction file as its prediction: this is the order in which your HPO method queried the tabular benchmark. 
Each row of the prediction file represents a single run of HPO (up to the allowed budget), and looks like this:

```
head -1 ./pred/so-en.predict
486 242 36 364 424 554 457 1 367 152 560 469 147 221 9 582 591 79 74 291 516 ...
# The HPO method first samples the hyperparameter configuration in row 486 of so-en.hyps, followed by 242 then 36.... The budget is 200, so 200 samples per line
```

The `./sample_code_submission` directory contains other examples:
- `hpo_model_bomo.py` for Bayesian Optimization (illustrates a sequential call to the `objection_function()` API)
- `hpo_model_cmaes.py` for CMA Evolutionary Strategy (illustrates a population-based method with calls to the `objection_function()` API)

To run these, you should replace the filename, e.g. `cp hpo_model_cmaes.py hpo_model.py`. These methods need some dependencies: `pip install cma scipy george` for local run, but for codalab submission the code needs to be self-contained.

(2) To score your HPO results, run the following:

```
# python scoring_program/evaluate.py $predictiondir $resultdir $referencedir 
# e.g. 
python scoring_program/evaluate.py ./pred ./ ../../datasets
```

`$predictiondir` is the result from step (1) ingestion program. `$resultdir` is where you evaluation result info will be stored. `$referencedir` points to the location in the tabular benchmark's `*.fronts` files, which labels the points that are Pareto-optimal.

We will be looking at the fixed-budget to pareto (fbp) metric, which measures how many Pareto solutions were found under a fixed budget (200), i.e. the more the merrier. Pareto solutions are those that are "optimal" on the speed-accuracy tradeoff curve. 
So the `$resultdir/scores.txt` might look like this: 

```
so-en_avg_pareto_with_fixed_budget: 1.8  # on average, your HPO method found 1.8 pareto points on the so-en dataset
so-en_standard_deviation: 1.03  # standard deviation of your HPO runs on so-en
sw-en_avg_pareto_with_fixed_budget: 3.9 # on average, your HPO method found 3.9 pareto points on the sw-en dataset
sw-en_standard_deviation: 1.52  # standard deviation of your HPO runs on sw-en
```

We desire HPO methods that find the maximum number of Pareto points in the tabular benchmark, as well as those that exhibit low standard deviation across multiple independent runs.

## To submit on Codalab

The above walkthrough is meant for local run. On codalab, we will be using the same ingestion and evaluation code, so you will just need to submit a zip file of your `$codedir`.

<b>IMPORTANT:</b> To ensure a smooth run on codalab, make sure you follow these instructions:

- There should be at least two files in your zip file: `hpo_model.py` and `metadata` (just include the same file in this `sample_code_submission/metadata`.)
- If you have dependencies, you can include those files in the same zip. If necessary, you can download them on-the-fly similar to the sample in `sample_code_submission/cmaes.py`, including the snippet:

```
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install('cma')
```

- Make sure the zip file does not contain a folder at the root, for example if we submit the code in the directory `sample_code_submission`, we might zip like this so that when things are unzipped, all the files are at the present working directory:

```
cd sample_code_submission && zip -r my_submission.zip . 
```

## Details about data and evaluation

There are many ways to evaluate, but we will focus on the following metric: the number of Pareto solutions discovered given a fixed budget. This is the fixed-budget-to-pareto (fbp) metric described in the <a href="https://www.cs.jhu.edu/~kevinduh/papers/zhang20benchmark.pdf">TACL paper</a>. Basically, your HPO method will be given a budget of 200 calls to the tabular benchmark API (i.e. it can sample at most 200 (x,f(x)) pairs. In the end, we will measure how many of these 200 samples (out of the all the hyperparameter settings in the tabular benchmark) are actually on the accuracy-speed Pareto frontier. Further, since some HPO methods incorporate some randomness inherently, we will make 10 independent runs and also record the standard deviation of the (fbp) metric.

The accuracy-speed objectives we will jointly optimize are:
- BLEU score (accuracy): Bigher is better. Ranges between 0 and 100.
- GPU inference time: Lower is better. This is the number of seconds it takes to translate a certain MT set.

On the Codalab leaderboard, your code will be run on two MT datasets (so-en, sw-en). The so-en dataset has 604 Transformer models (with different hyperparameter configurations), and the sw-en dataset has 767 hyperparameter configurations. This repo contains four other datasets for you to run locally, but note they have fewer models (less than 200), so we recommend reducing the budget to 50 when testing out your method. The blind test data during the evaluation phase will be similar to so-en and sw-en in characteristics: similar number of models in the tabular benchmark, and same budget of 200 calls. Specifically, the blind test set will feature a different language-pair for MT and different GPU used to measure decoding time, but same hyperparameter space and same Transformer training pipeline. We will allow participants to explore HPO Transfer Learning methods, meaning that data from the six public MT datasets here can be exploited for improving HPO on the new test MT dataset.

The Transformer hyperparameters explored in this competition are as follows:
- bpe_symbols: Number of subword units for the input/output vocabulary. This interacts with both speed and accuracy (BLEU) in unexpected ways, i.e. for speed, fewer units means faster output softmax layer, but extend sequence length; for accuracy, fewer units mean less paramaters to fit but less correspond to "full words"
- num_layers: Number of encoder/decoder layers in Transformer. More layers usually mean higher accuracy, at the risk of speed decrease.
- num_embed: Number of word embedding dimension
- transformer_feed_forward_num_hidden: Number of hidden units in feedward layer of transformer blocks.
- transformer_attention_heads: Number of heads in the transformer block.
- initial_learning_rate: The initial learning rate for ADAM gradient-based training; this can affect accuracy significantly, but not speed

You may want to scale the hyperparameter features in a range suitable for your HPO method; see code example of `transform_lst` in CMA-ES at `sample_code_submission/cmaes.py`. 