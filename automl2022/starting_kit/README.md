This is the starting kit for the AutoML2022 competition

(1) To run a hyperparameter optimization (HPO) and generate predictions, run the following: 
```
python ingestion_program/ingestion.py ../../datasets $predictiondir ./ingestion_program ./sample_code_submission
```

`$predictiondir` is the directory where your model predictions will be stored.
Note the sample_code_submission directory includes the file `hpo_model.py`. Currently its set as a random search baseline.
To implement your own HPO method, please modify the run() function. 

Specifically, the ingestion program will read run hpo_model.run() using different benchmark datasets.
Each time, it will generate a sample sequence file as its prediction: this is the order in which your HPO method queried the tabular benchmark. 

The directory contains other examples:
- `hpo_model_bomo.py` for Bayesian Optimization (illustrates a sequential call to the `objection_function()` API)
- `hpo_model_cmaes.py` for CMA Evolutionary Strategy (illustrates a population-based method with calls to the `objection_function()` API)

To run these, you should replace the filename, e.g. `cp hpo_model_cmaes.py hpo_model.py`. These methods need some dependencies: `pip install cma scipy george` for local run, but for codalab submission the code needs to be self-contained.

(2) To score your HPO results, run the following:
```
python scoring_program/evaluate.py $predictiondir $resultdir ../../datasets 
```

`$predictiondir` is the result from step (1) ingestion program. `$resultdir` is where you evaluation result info will be stored. 

We will be looking at the fixed-budget (fb) metric, which measures how many Pareto solutions were found under a fixed budget.

TODO: From the above command, the sample sequence file and prediction results are stored in the current directory (see `*.predict` and `*.result`). This may be changed. 
