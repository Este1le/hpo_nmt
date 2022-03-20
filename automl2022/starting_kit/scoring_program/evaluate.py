"""
Wrapper script for evaluation
"""

import os
import sys

input_dir = sys.argv[1]
output_dir = sys.argv[2]

total_budget = 200
for dataset in ['so-en','sw-en']:

    predict_file = os.path.join(output_dir, "%s.predict"%dataset)
    reference_file = os.path.join(input_dir, "%s.fronts"%dataset)
    result_file = os.path.join(output_dir, "%s.result"%dataset)

    # Run evaluation script on predict_file, saving results in result_file
    os.system("python scoring_program/eval_multiple.py -i 0 -b %d -s %s -f %s > %s" % (total_budget, predict_file, reference_file, result_file))

    # Parse result_file
    with open(result_file,'r') as RESULT:
        results = RESULT.readlines()[-1].split()
        fb = float(results[1].rstrip(','))
        stddev = round(float(results[3]),2)
        print(f"{dataset}: avg #pareto discovered given fixed-budget(fb): {fb} (stddev: {stddev} )")