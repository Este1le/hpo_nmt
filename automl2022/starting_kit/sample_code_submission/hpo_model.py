"""
Hyperparameter Optimization implementation example
Random Search
"""

import random


def run(bench, budget):
    """Executes one Hyperparameter Optimization (HPO) run.

    :param TabularBenchmark bench: an object instantiated by ingestion.py
        that tracks the hyperparameters and metrics of a dataset
    :param int budget: max calls to bench.objective_function() API
    :return nothing but saves the order of API queries in a file for evaluation
    """

    # Get the set of hypeparameter configuration space possible in this benchmark
    cs = bench.get_configuration_space()

    ##############################################################################
    # Begin implementation
    ##############################################################################

    # Randomly sample these configurations until budget filled
    random.shuffle(cs)
    for i in range(budget):
        print("Sample:", bench.objective_function(cs[i]))

        # if you need to convert the string in cs[i] into a feature vector, do:
        hyperparam_vector = [float(feature) for feature in cs[i].split('\t')]

    ##############################################################################
    # End implementation
    ##############################################################################
    # This needs to be called at the end of a run
    bench.done()
