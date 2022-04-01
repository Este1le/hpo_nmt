#!/usr/bin/env python

# Usage: python ingestion.py input_dir output_dir ingestion_program_dir submission_program_dir

# AS A PARTICIPANT, DO NOT MODIFY THIS CODE.
#
# This is the "ingestion program" written by the organizers.
# This program also runs on the challenge platform to test your code.

import sys
import os
import logging
from datetime import datetime
from warnings import catch_warnings 
from tabular_benchmark import TabularBenchmark


if __name__=="__main__":

    # Setup paths
    input_dir = os.path.abspath(sys.argv[1])
    output_dir = os.path.abspath(sys.argv[2])
    program_dir = os.path.abspath(sys.argv[3])
    submission_dir = os.path.abspath(sys.argv[4])

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s %(levelname)s %(filename)s - %(message)s')
    logger = logging.getLogger('ingestion')
    logger.info("=== Starting Ingestion Program ===")
    logger.info("Getting Data from input_dir: " + input_dir)
    logger.info("Putting Predictions in output_dir: " + output_dir)
    logger.info("Using Ingestion code from program_dir: " + program_dir)
    logger.info("Using HPO code from submission_dir: " + submission_dir)

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except Exception as e:
            logger.exception("Cannot create {output_dir}: {e}")

    sys.path.append(program_dir)
    sys.path.append(submission_dir)
    import hpo_model as hpo_model

    # For Codalab leaderboard, we will only compare on the so-en and sw-en datasets. 
    # Feel free to try other dataset in your local run (e.g. en-ja, ja-en, ru-en, zh-en)
    total_budget = 200
    total_runs = 10
    accumulated_time = 0.0
    logger.info(f"For each dataset, HPO will run independently for {total_runs} times, with a budget of {total_budget} each.")

    # 2. Run HPO code over several datasets
    for dataset in ['so-en', 'sw-en']:

        # We'll have multiple runs of HPO to measure the variance of methods
        for run_number in range(total_runs):
            start_time = datetime.now()
            logger.info(f"Dataset: {dataset} Run: {run_number} started.")

            # This object contains the table of hyperparameter settings and their objective values
            bench = TabularBenchmark(input_dir, output_dir, dataset, total_budget, run_number)

            # This function requires the hpo_model.py file, provided by the participant
            hpo_model.run(bench, total_budget)

            # Record run time
            elapsed_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Dataset: {dataset} Run: {run_number} completed in {elapsed_time} seconds")
            accumulated_time += elapsed_time
    
    logger.info(f"=== Ingestion and HPO runs completed in {accumulated_time} seconds ===")

sys.exit(0)
