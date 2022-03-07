"""
Tabular Benchmark Manager
"""

import os

class TabularBenchmark():
    def __init__(self, mt_dataset):
        x_file = os.path.join('../datasets', "%s.hyps" %mt_dataset)
        y_file = os.path.join('../datasets', "%s.fronts" %mt_dataset)
        evals_file = os.path.join('../datasets', "%s.evals" %mt_dataset)

        self.mt_dataset = mt_dataset
        self._budget_used = 0
        self.config = []
        with open(x_file, 'r') as X, open (y_file, 'r') as Y:
            for line in X:
                self.config.append(line.rstrip().split('\t'))
            y = [yy.rstrip() for yy in Y]

        self.evals = {}
        with open(evals_file, 'r') as E:
            for id, line in enumerate(E):
                dev_bleu, dev_gpu_time = line.rstrip().split()[0:2]
                config_str = " ".join(self.config[id])
                # TODO: standize eval rounding
                self.evals[config_str] = {'bleu_score':float(dev_bleu),
                                        'decode_time':int(float(dev_gpu_time)),
                                        'id':id}
        
        # TODO: improve how we collect samples for results evaluation
        self.sample_sequence_fid = open('tmp.ss', 'w')


    @property
    def budget_used(self):
        # TODO: protect this variable in some way?
        return self._budget_used

    def get_configuration_space(self):
        return self.config

    def objective_function(self, config):
        config_str = " ".join(config)
        if config_str in self.evals:
            self._budget_used += 1
            evals = self.evals[config_str]
            self.sample_sequence_fid.write("%d "%evals['id'])
            return evals
        else:
            return {'bleu_score':None, 'decode_time':None, 'id':None}


    # TODO: this should be re-written to directly call on eval_multiple.py as module
    def evaluate_result(self):
        self.sample_sequence_fid.write('\n')
        self.sample_sequence_fid.close()
        os.system("python ../scripts/eval_multiple.py -i 3 -s tmp.ss -f ../datasets/%s.fronts" % self.mt_dataset)


if __name__ == '__main__':

    # example
    bench = TabularBenchmark('en-ja')
    cs = bench.get_configuration_space()
    print(bench.objective_function(cs[0]))