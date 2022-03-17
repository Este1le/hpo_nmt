"""
Hyperparameter Optimization implementation example - Multi-Objective Bayesian Optimization.
"""

from bomo import BOMO

def run(bench, budget):

    cs = bench.get_configuration_space()
    bomo = BOMO(cs)

    for _ in range(budget):
        i = bomo.sample()
        sample = bench.objective_function(cs[i])
        print("Sample:", sample)
        bomo.fit_predict(sample)

    bench.done()