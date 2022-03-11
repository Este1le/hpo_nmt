import os
import sys

input_dir = sys.argv[1]
output_dir = sys.argv[2]


for dataset in ['en-ja']:
    predict_file = os.path.join(output_dir, "%s.predict"%dataset)
    reference_file = os.path.join(input_dir, "%s.fronts"%dataset)
    result_file = os.path.join(output_dir, "%s.result"%dataset)
    os.system("python scoring_program/eval_multiple.py -i 3 -s %s -f %s > %s" % (predict_file, reference_file, result_file))
