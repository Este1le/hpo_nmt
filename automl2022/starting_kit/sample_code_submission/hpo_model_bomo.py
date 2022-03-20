"""
Hyperparameter Optimization implementation example: 
Multi-Objective Bayesian Optimization.
"""

from bomo import BOMO

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
    bomo = BOMO(cs)

    for _ in range(budget):
        i = bomo.sample()
        sample = bench.objective_function(cs[i])
        print("Sample:", sample)
        bomo.fit_predict(sample)

    ##############################################################################
    # End implementation
    ##############################################################################
    # This needs to be called at the end of a run
    bench.done()
