"""
An Implementation of Pareto based multiobjective CMA-ES (covariance matrix adaptation-evolution strategy).
Reference: AUTOMATED STRUCTURE DISCOVERY AND PARAMETER TUNING OF NEURAL NETWORK LANGUAGE MODEL BASED ON EVOLUTION STRATEGY,
           Tomohiro Tanaka, Takafumi Moriya, Takahiro Shinozaki, Shinji Watanabe, Takaaki Hori, Kevin Duh, 2016
"""
import math
import cma
import random
import numpy as np
from pareto import pareto
from scipy import spatial

transform_lst = [lambda x: (x-10000)/40000,
                 lambda x: (x-2)/2,
                 lambda x: (math.log(x, 2)-8)/2,
                 lambda x: (x-1024)/1024,
                 lambda x: (x-8)/8,
                 lambda x: (x-0.0003)/0.0007]

class CMAES:
    def __init__(self, cs, pop=5, num_obj=2):
        cs = np.array(cs)
        cs = cs.astype(np.float)
        self.cs = []
        for c in cs:    
            self.cs.append([transform_lst[i](c[i]) for i in range(len(c))])
        self.label = np.full(len(self.cs), False)
        init_id = random.randint(0, len(self.cs))
        init_vec = self.cs[init_id]
        self.es = cma.CMAEvolutionStrategy(init_vec, 0.1, {
            'seed': 1,
            'popsize': pop,
        })
        self.cs = np.array(self.cs)
        self.cur_pool = []
        self.eval = np.zeros((len(cs), num_obj))

    def sample(self):
        raw_c = self.es.ask()
        for rc in raw_c:
            new_id = self._find_neighbor(rc)
            self.label[new_id] = True
            self.cur_pool.append(new_id)
        return self.cur_pool
    
    def fit_predict(self, raw_evals):
        evals = []
        for ri in range(len(self.cur_pool)):
            new_eval = self._transform_eval(raw_evals[ri])
            evals.append(new_eval)
        evals = np.array(evals)
        ranking = -np.array(pareto(evals.T, [-1,-1]))
        self.es.tell(self.cs[self.cur_pool], ranking) # cmaes: fmin
        self.cur_pool = []
    
    def _find_neighbor(self, x):
        dis = np.full(len(self.cs), np.inf)
        for u in np.where(~self.label)[0]:
            dis[u] = spatial.distance.cosine(x, self.cs[u])
        id = np.nanargmin(dis)
        return id

    def _transform_eval(self, eval):
        y1 = -eval['bleu_score'] # -BLEU, smaller is better
        y2 = eval['decode_time'] # decoding_time, smaller is better
        return [y1, y2]
    
