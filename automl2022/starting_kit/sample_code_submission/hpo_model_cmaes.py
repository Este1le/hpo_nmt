"""
Hyperparameter Optimization implementation example - CMA-ES
"""

from cmaes import CMAES

def run(bench, budget):

    cs = bench.get_configuration_space()
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
        
    bench.done()

