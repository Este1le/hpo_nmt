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
    #import hpo_model_bomo as hpo_model
    #import hpo_model_cmaes as hpo_model

    for dataset in ['en-ja']:
        bench = TabularBenchmark(input_dir, output_dir, dataset)
        hpo_model.run(bench, 50)

