"""
Hyperparameter Optimization implementation example - Random Search
"""

import random
#from tabular_benchmark import TabularBenchmark

# TODO: better documentation
def run(bench, budget):

    cs = bench.get_configuration_space()
    random.shuffle(cs)

    for i in range(budget):
        print("Sample:", bench.objective_function(cs[i]))

    bench.done()

