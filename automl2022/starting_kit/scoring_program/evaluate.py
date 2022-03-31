"""
Wrapper script for evaluation
"""

import os
import sys
import statistics

if len(sys.argv) == 4:
    prediction_dir = sys.argv[1]
    output_dir = sys.argv[2]
    reference_dir = sys.argv[3]
elif len(sys.argv) == 3:
    prediction_dir = os.path.join(sys.argv[1],'res/')
    output_dir = sys.argv[2]
    reference_dir = os.path.join(sys.argv[1],'ref/')
else:
    print("Expects either:\n (a) evaluate.py predict_dir output_dir reference_dir\n (b) evaluate.py input_dir output_dir")
    sys.exit(1)

total_budget = 200
score_file = os.path.join(output_dir, "scores.txt")
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
        for i in range(total_budget):
            if pred[i] in reference:
                fb += 1
        fb_runs.append(fb)
    fb_ave, fb_std = statistics.mean(fb_runs), statistics.stdev(fb_runs)
    fb_std = round(fb_std, 2)

    # Save results in score_file
    with open(score_file, 'a') as f:
        f.write("{0}_avg_pareto_with_fixed_budget: {1}\n".format(dataset,fb_ave))
        f.write("{0}_standard_deviation: {1}\n".format(dataset, fb_std))

    print(f"{dataset}: avg #pareto discovered given fixed-budget(fb): {fb_ave} (stddev: {fb_std} )")