"""
Hyperparameter Optimization implementation example - Random Search
"""

import random
from tabular_benchmark import TabularBenchmark

bench = TabularBenchmark('en-ja')
cs = bench.get_configuration_space()
random.shuffle(cs)

budget = 53
for i in range(budget):
    print("Sample:", bench.objective_function(cs[i]))

bench.evaluate_result()

