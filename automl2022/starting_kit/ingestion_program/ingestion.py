#!/usr/bin/env python

# Usage: python ingestion.py input_dir output_dir ingestion_program_dir submission_program_dir

# AS A PARTICIPANT, DO NOT MODIFY THIS CODE.
#
# This is the "ingestion program" written by the organizers.
# This program also runs on the challenge platform to test your code.

import sys
import os
from tabular_benchmark import TabularBenchmark

if __name__=="__main__":

    input_dir = os.path.abspath(sys.argv[1])
    output_dir = os.path.abspath(sys.argv[2])
    program_dir = os.path.abspath(sys.argv[3])
    submission_dir = os.path.abspath(sys.argv[4])

    print("Using input_dir: " + input_dir)
    print("Using output_dir: " + output_dir)
    print("Using program_dir: " + program_dir)
    print("Using submission_dir: " + submission_dir)

    sys.path.append(program_dir)
    sys.path.append(submission_dir)
    import hpo_model as hpo_model

    # For Codalab leaderboard, we will only compare on the so-en and sw-en datasets. 
    # Feel free to try other dataset in your local run (e.g. en-ja, ja-en, ru-en, zh-en)
    total_budget = 200
    total_runs = 10
    for dataset in ['so-en', 'sw-en']:

        # We'll have multiple runs of HPO to measure the variance of methods
        for run_number in range(total_runs):

            # This object contains the table of hyperparameter settings and their objective values
            bench = TabularBenchmark(input_dir, output_dir, dataset, total_budget, run_number)

            # This function requires the hpo_model.py file, provided by the participant
            hpo_model.run(bench, total_budget)

