"""
Hyperparameter Optimization implementation example:
CMA-ES
"""

from cmaes import CMAES

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
    popsize=5
    cmaes = CMAES(cs, pop=popsize)

    for i in range(int(budget/popsize)):
        pool = cmaes.sample()
        evals = []
        for i in pool:
            eval = bench.objective_function(cs[i])
            evals.append(eval)
            print("Sample:", eval)
        cmaes.fit_predict(evals)

    ##############################################################################
    # End implementation
    ##############################################################################
    # This needs to be called at the end of a run
    bench.done()
