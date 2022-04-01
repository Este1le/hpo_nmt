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

The above walkthrough is meant for local run. On codalab, we will be using the same ingestion and evaluation code, so you will just need to submit a zip file of your `$codedir`. Make sure the zip file does not contain a folder at the root, for example if we submit the code in the directory `sample_code_submission`, we might zip like this:

```
cd sample_code_submission && zip -r my_submission.zip . 
```
