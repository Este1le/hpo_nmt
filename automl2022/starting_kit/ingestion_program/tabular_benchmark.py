"""
Tabular Benchmark Manager
"""

import os

class TabularBenchmark():
    def __init__(self, input_dir, output_dir, mt_dataset, total_budget, run_number):
        hyps_file = os.path.join(input_dir, "%s.hyps" %mt_dataset)
        evals_file = os.path.join(input_dir, "%s.evals" %mt_dataset)

        self._total_budget = total_budget
        self._budget_used = 0
        self.config = []

        with open(hyps_file, 'r') as H:
            for line in H:
                self.config.append(line.rstrip())

        self.evals = {}
        with open(evals_file, 'r') as E:
            for id, line in enumerate(E):
                dev_bleu, dev_gpu_time = line.rstrip().split()[0:2]
                self.evals[self.config[id]] = {'bleu_score':float(dev_bleu),
                                        'decode_time':float(dev_gpu_time),
                                        'id':id}
        
        if run_number == 0:
            self.sample_sequence_fid = open(os.path.join(output_dir,'%s.predict'%mt_dataset), 'w')
        else:
            self.sample_sequence_fid = open(os.path.join(output_dir,'%s.predict'%mt_dataset), 'a')


    @property
    def budget_used(self):
        """Returns budget used so far
        """
        return self._budget_used

    @property
    def total_budget(self):
        """Returns total budgeted allowed (i.e. number of calls to objective_function) for this benchmark
        """
        return self._total_budget

    def get_configuration_space(self):
        """Returns a list of strings, representing the hyperparameter configuration space of this tabular benchmark
        e.g. cs = bench.get_configuration_space()
             cs[i] is the i-th hyperparameter vector, represented as a tab-delimited string
             like this: '2000.0 \t 2.0 \t 1024.0 \t 1024.0 \t 16.0 \t 0.0003'
        """
        return self.config

    def objective_function(self, config_str):
        """Returns the accuracy/speed metric of a given hyperparameter configuration
        
        :param config_str string: this represents the configuration, should be same as given in self.config[id]
        :return dictionary with keys 'bleu_score', 'decode_time', and 'id' of the queried hyperparameter configuration
        if the budget is reached, or the config_str does not exist in table, then dictionary will contain None values 
        """

        # if max budget reached, return nothing
        if self._budget_used >= self._total_budget:
            return {'bleu_score':None, 'decode_time':None, 'id':None}

        # return speed/accuracy metrics as  dictionary, if the config_str exists
        if config_str in self.evals:
            self._budget_used += 1
            evals = self.evals[config_str]
            # write results in prediction file for evaluation later
            self.sample_sequence_fid.write("%d "%evals['id'])
            return evals
        else:
            return {'bleu_score':None, 'decode_time':None, 'id':None}


    def done(self):
        self.sample_sequence_fid.write('\n')
        self.sample_sequence_fid.close()
