"""
Wrapper script for evaluation
"""

import os
import sys
import statistics
import logging


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s %(levelname)s %(filename)s - %(message)s')
logger = logging.getLogger('evaluate')

# This script can actually be called in two different ways
# The one with 3 arguments is intended for local run and the one with 2 arguments is for codalab
if len(sys.argv) == 4:
    # local run
    prediction_dir = sys.argv[1]
    output_dir = sys.argv[2]
    reference_dir = sys.argv[3]
elif len(sys.argv) == 3:
    # codalab run, hardcoding some paths that are passed under the hood
    prediction_dir = os.path.join(sys.argv[1],'res/')
    output_dir = sys.argv[2]
    reference_dir = os.path.join(sys.argv[1],'ref/')
else:
    logger.error("Expects either:\n (a) evaluate.py predict_dir output_dir reference_dir\n (b) evaluate.py input_dir output_dir")
    sys.exit(1)


logger.info("=== Starting Scoring/Evaluate Program ===")
logger.info(f"Using predictions from prediction_dir: {prediction_dir}")
logger.info(f"Using labels for Pareto points from reference_dir: {reference_dir}")
logger.info(f"Storing evaluation results in output_dir: {output_dir}")

total_budget = 200
score_file = open(os.path.join(output_dir, "scores.txt"),'w')

dataset_number = 1
for dataset in ['so-en','sw-en']:

    predict_file = os.path.join(prediction_dir, "%s.predict"%dataset)
    reference_file = os.path.join(reference_dir, "%s.fronts"%dataset)

    prediction = []
    with open(predict_file) as f:
        for l in f.readlines():
            prediction.append([int(_) for _ in l.split()])
    
    reference = []
    with open(reference_file) as f:
        lines = f.readlines()
    for i in range(len(lines)):
        if int(lines[i]) == 1:
            reference.append(i)

    # fb metric: Fix the budget of function evaluations and measure 
    # the number of Pareto-optimal points obtained.
    fb_runs = []
    for pred in prediction:
        fb =  0
        visited_pred = []
        for i in range(total_budget):
            if pred[i] in reference and pred[i] not in visited_pred:
                fb += 1
                visited_pred.append(pred[i])
        fb_runs.append(fb)
    fb_ave, fb_std = statistics.mean(fb_runs), statistics.stdev(fb_runs)
    fb_std = round(fb_std, 2)

    # Save results in score_file
    score_file.write("set{0}_avg_pareto_with_fixed_budget: {1}\n".format(dataset_number,fb_ave))
    score_file.write("set{0}_standard_deviation: {1}\n".format(dataset_number, fb_std))

    logger.info(f"set{dataset_number} {dataset}: average #pareto solutions discovered given fixed-budget of {total_budget} (fbp): {fb_ave}")
    logger.info(f"set{dataset_number} {dataset}: standard deviation of #pareto over {len(fb_runs)} runs: {fb_std}")
    dataset_number += 1

score_file.close()
sys.exit(0)
