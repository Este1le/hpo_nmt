import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate HPO methods on multi-objective optimization.")
    parser.add_argument("--sample-sequence", "-s", type=str, help="The path to a file of sampling sequences."
                                                                    "Each line contains numbers separated by space."
                                                                    "A sampling sequence is a list of sample ids"
                                                                    "sorted by sampling order."
                                                                    "Multiple sampling sequences with different initial"
                                                                    "samples are allowed, in this case, there are "
                                                                    "multiple lines in the file. "
                                                                    "Example: examples/example.ss")
    parser.add_argument("--fronts", "-f", type=str, help="The path to a file of Pareto-optimal samples."
                                                         "Each line contains a number 0 or 1."
                                                         "1 indicates Pareto-optimal sample."
                                                         "Example: examples/example.fronts")
    parser.add_argument("--num-init", "-i", type=int, help="Number of initial samples.")
    parser.add_argument("--budget", "-b", default=50, help="Runtime budget for fb.")
    return parser.parse_args()

def fto(sequence, fronts, num_init):
    '''
    Measure the runtime to get one Pareto point.
    :param sequence: A list of sample ids sorted by sampling order.
    :param fronts: A list of Pareto-optimal sample ids.
    :param num_init: Number of initial samples.
    :return: fto runtime.
    '''
    runtime = 0
    for i in range(len(sequence)):
        runtime += 1
        if sequence[i] in fronts:
            break
    if runtime < num_init:
        return num_init
    else:
        return runtime



def fta(sequence, fronts, num_init):
    '''
    Measure the runtime to find all points on the Pareto front.
    :param sequence: A list of sample ids sorted by sampling order.
    :param fronts: A list of Pareto-optimal sample ids.
    :param num_init: Number of initial samples.
    :return: fta runtime.
    '''
    runtime = 0
    for i in range(len(sequence)):
        runtime += 1
        if sequence[i] in fronts:
            fronts = [f for f in fronts if f!=sequence[i]]
            if len(fronts) == 0:
                break
    if runtime < num_init:
        return num_init
    else:
        return runtime

def fb(sequence, fronts, budget):
    '''
    Fix the budget of function evaluations and measure the number of Pareto-optimal points obtained.
    :param sequence: A list of sample ids sorted by sampling order.
    :param fronts: A list of Pareto-optimal sample ids.
    :param budget: runtime budget.
    :return: number of Pareto-optimal points found.
    '''
    nf = 0
    for i in range(budget):
        if sequence[i] in fronts:
            nf += 1
    return nf

if __name__ == "__main__":
    args = get_args()
    sample_sequence = args.sample_sequence
    ff = args.fronts
    num_init = args.num_init
    budget = args.budget

    # Read sampling sequences
    with open(sample_sequence) as f:
        lines = f.readlines()
    ss = []
    for l in lines:
        ss.append([int(_) for _ in l.split()])

    # Read fronts
    with open(ff) as f:
        lines = f.readlines()
    fronts = []
    for i in range(len(lines)):
        if int(lines[i]) == 1:
            fronts.append(i)

    # Get mean and standard deviation of fto, fta, fb runtime
    fto_rt = []
    fta_rt = []
    fb_rt = []
    for sequence in ss:
        fto_rt.append(fto(sequence, fronts, num_init))
        fta_rt.append(fta(sequence, fronts, num_init))
        fb_rt.append(fb(sequence, fronts, budget))
    fto_ave, fto_std = np.average(fto_rt), np.std(fto_rt)
    fta_ave, fta_std = np.average(fta_rt), np.std(fta_rt)
    fb_ave, fb_std = np.average(fb_rt), np.std(fb_rt)

    # Print out the results
    res = "ave: {0}, std: {1}"
    print("fto")
    #print(fto_rt)
    print(res.format(fto_ave, fto_std))

    print("fta")
    #print(fta_rt)
    print(res.format(fta_ave, fta_std))

    print("fb")
    #print(fb_rt)
    print(res.format(fb_ave, fb_std))

