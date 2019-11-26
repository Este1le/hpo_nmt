import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate HPO methods on single-objective optimization.")
    parser.add_argument("--sample-sequence", "-s", type=str, help="The path to a file of sampling sequences."
                                                                    "Each line contains numbers separated by space."
                                                                    "A sampling sequence is a list of sample ids"
                                                                    "sorted by sampling order."
                                                                    "Multiple sampling sequences with different initial"
                                                                    "samples are allowed, in this case, there are "
                                                                    "multiple lines in the file. "
                                                                    "Example: examples/example.ss")
    parser.add_argument("--evals", "-e", type=str, help="The path to a file of model evaluations."
                                                          "This file should contain a list of numbers (e.g. BLEU) --"
                                                          "one line a number, and can be indexed by sample ids."
                                                          "Example: examples/example.bleu")
    parser.add_argument("--num-init", "-i", type=int, help="Number of initial samples.")
    parser.add_argument("--tolerance", "-t", default=0.5, help="Tolerance of performance degradation for ftc.")
    parser.add_argument("--budget", "-b", default=20, help="Runtime budget for fb.")
    parser.add_argument("--minimization", "-m", action="store_true", help="Maximization or minimization,"
                                                                          "e.g. for BLEU, it is maximization."
                                                                          "Default is maximization.")
    return parser.parse_args()

def ftb(sequence, best, num_init):
    '''
    Fix the quality indicator value to the best value (e.g. oracle BLEU) in the dataset and
    measure runtime to reach this target, where runtime is defined as the number of evaluated samples.
    :param sequence: A list of sample ids sorted by sampling order.
    :param best: Id of the sample with best quality indicator value.
    :param num_init: Number of initial samples.
    :return: ftb runtime.
    '''
    runtime =  sequence.index(best) + 1
    if runtime < num_init:
        return num_init
    else:
        return runtime


def ftc(sequence, best, num_init, evals, tolerance):
    '''
    Measure the runtime to reach a target that is slightly less than the oracle best.
    :param sequence: A list of sample ids sorted by sampling order.
    :param best: Id of the sample with best quality indicator value.
    :param num_init: Number of initial samples.
    :param evals: A list of quality indicator values of the samples indexed by sample ids.
    :param tolerance: Tolerance of performance degradation for ftc.
    :return: ftc runtime.
    '''
    for i in range(len(sequence)):
        e = evals[sequence[i]]
        if e >= evals[best]-tolerance:
            if i < num_init:
                return num_init
            else:
                return i + 1

def fb(sequence, best, evals, budget):
    '''
    Fix the budget of function evaluations and measure the difference between
    the best quality indicator value in the dataset vs.
    the maximum value achieved by systems queried by the HPO method.
    :param sequence: A list of sample ids sorted by sampling order.
    :param best: Id of the sample with best quality indicator value.
    :param num_init: Number of initial samples.
    :param evals: A list of quality indicator values of the samples indexed by sample ids.
    :param budget: runtime budget.
    :return: The difference.
    '''
    cur = evals[sequence[0]]
    for i in range(1, len(sequence[:budget])):
        e = evals[sequence[i]]
        if e > cur:
            cur = e
    return evals[best] - cur

if __name__ == "__main__":
    args = get_args()
    sample_sequence = args.sample_sequence
    evals = args.evals
    num_init = args.num_init
    tolerance = args.tolerance
    budget = args.budget

    # Read sampling sequences
    with open(sample_sequence) as f:
        lines = f.readlines()
    ss = []
    for l in lines:
        ss.append([int(_) for _ in l.split()])

    # Read evals
    with open(evals) as f:
        lines = f.readlines()
    evals = []
    for l in lines:
        evals.append(float(l))

    # Get best eval
    if args.minimization:
        best = np.argmin(evals)
    else:
        best = np.argmax(evals)

    # Get mean and standard deviation of ftb, ftc, fb runtime
    ftb_rt = []
    ftc_rt = []
    fb_rt = []
    for sequence in ss:
        ftb_rt.append(ftb(sequence, best, num_init))
        ftc_rt.append(ftc(sequence, best, num_init, evals, tolerance))
        fb_rt.append(fb(sequence, best, evals, budget))
    ftb_ave, ftb_std = np.average(ftb_rt), np.std(ftb_rt)
    ftc_ave, ftc_std = np.average(ftc_rt), np.std(ftc_rt)
    fb_ave, fb_std = np.average(fb_rt), np.std(fb_rt)

    # Print out the results
    res = "ave: {0}, std: {1}"
    print("ftb")
    #print(ftb_rt)
    print(res.format(ftb_ave, ftb_std))

    print("ftc")
    #print(ftc_rt)
    print(res.format(ftc_ave, ftc_std))

    print("fb")
    #print(fb_rt)
    print(res.format(fb_ave, fb_std))

