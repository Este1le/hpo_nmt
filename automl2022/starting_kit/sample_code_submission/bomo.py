"""
An Implementation of Multi-Objective Bayesian Optimization using EHVI (expected hypervolume improvement).
Reference: Hypervolume-based Expected Improvement: Monotonicity Properties and Exact Computation, Emmerich et al. 2011
"""
import sys
import subprocess
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
install('george')
install('scipy')
install('numpy')
import numpy as np
from scipy.stats import norm
import george
from pareto import pareto
import math

transform_lst = [lambda x: (x-10000)/40000,
                 lambda x: (x-2)/2,
                 lambda x: (math.log(x, 2)-8)/2,
                 lambda x: (x-1024)/1024,
                 lambda x: (x-8)/8,
                 lambda x: (x-0.0003)/0.0007]

class BOMO():
    def __init__(self, cs, init_num=3, num_obj=2):
        self.cs = []
        for c in cs:
            cf = [float(f) for f in c.split('\t')]
            self.cs.append([transform_lst[i](cf[i]) for i in range(len(cf))])
        self.cs = np.array(self.cs)
        self.label = np.full(len(cs), False)
        self.init_num = init_num
        self.eval = np.zeros((len(cs), num_obj))
        self.num_obj = num_obj
        self.cur_id = -1
        n_dims = len(self.cs[0])
        kernel = george.kernels.Matern52Kernel(np.ones(n_dims), ndim=n_dims)
        self.gp = george.GP(kernel)

    def sample(self):
        if np.sum(self.label) < self.init_num:
            self.cur_id = np.random.choice(np.where(~self.label)[0])
        self.label[self.cur_id] = True
        return self.cur_id

    def fit_predict(self, new_eval):
        new_eval = self._transform_eval(new_eval)
        self.eval[self.cur_id] = new_eval
        evals = self.eval[self.label, :]
        if np.sum(self.label) >= self.init_num:
            ranking = np.array(pareto(evals.T, [-1,-1]))
            current_front_ids, = np.where(ranking==max(ranking))
            P = evals[current_front_ids, :]

            eval_preds = np.zeros(self.eval.shape)
            eval_vars = np.zeros(self.eval.shape)
            for k in range(self.num_obj):
                eval_preds[:, k], eval_vars[:, k] = self._gp_predict(evals[:, k])
            
            r = np.max(np.vstack((np.max(eval_preds, 0), np.max(evals, 0))), 0) + 1
            ehvis = np.full(len(self.cs), -np.inf)
            for u in np.where(~self.label)[0]:
                ehvis[u] = self._ehvi(P, r, eval_preds[u, :], np.sqrt(eval_vars[u, :]))
            self.cur_id = np.argmax(ehvis)

    def _gp_predict(self, evals):
        self.gp.compute(self.cs[self.label], yerr=1e-3) 
        pred_mean, pred_var = self.gp.predict(evals, self.cs, return_var=True)
        pred_var = np.absolute(pred_var)
        return pred_mean, pred_var
    
    def _transform_eval(self, eval):
        y1 = -eval['bleu_score'] # -BLEU, smaller is better
        y2 = eval['decode_time'] # decoding_time, smaller is better
        return [y1, y2]

    def _exipsi(self, a, b, mu, sigma):
        result1 = sigma * norm.pdf((b-mu)/sigma)
        result2 = (a-mu) * norm.cdf((b-mu)/sigma)
        return result1 + result2

    def _hvolume2d(self, P, x):
        # P: k * 2, x: (x1, x2)
        S = np.sort(P, axis=0)
        h = 0
        if (P.shape[0]!=0):
            for i in range(0, P.shape[0]):
                if i==0:
                    h += (x[0]-S[i][0]) * (x[1]-S[i][1])
                else:
                    h += (x[0]-S[i][0]) * (S[i-1][1]-S[i][1])
        return h
    
    def _ehvi(self, P, r, mu, s):
        # P: k*2, r:(r1, r2), mu:(mu1, mu2), s:(s1, s2)
        S = np.sort(P, axis=0)
        k = S.shape[0]
        c1 = np.sort(S[:,0])
        c2 = np.sort(S[:,1])
        res = 0
        for i in range(-1, k):
            for j in range(-1, k-i-1):
                if j==-1:
                    fMax2 = r[1]
                else:
                    fMax2 = c2[k-1-j]
                if i==-1:
                    fMax1 = r[0]
                else:
                    fMax1 = c1[k-1-i]
                if j==-1:
                    cL1 = float('-inf')
                else:
                    cL1 = c1[j]
                if i==-1:
                    cL2 = float('-inf')
                else:
                    cL2 = c2[i]
                if j==k-1:
                    cU1 = r[0]
                else:
                    cU1 = c1[j+1]
                if i==k-1:
                    cU2 = r[1]
                else:
                    cU2 = c2[i+1]

                SM = []
                for m in range(k):
                    if (cU1 <= S[m][0]) and (cU2 <= S[m][1]):
                        SM.append([S[m][0], S[m][1]])
                SM = np.array(SM)
                sPlus = self._hvolume2d(SM, [fMax1, fMax2])
                Psi1 = self._exipsi(fMax1, cU1, mu[0], s[0]) - self._exipsi(fMax1, cL1, mu[0], s[0])
                Psi2 = self._exipsi(fMax2, cU2, mu[1], s[1]) - self._exipsi(fMax2, cL2, mu[1], s[1])
                GaussCDF1 = norm.cdf((cU1-mu[0])/s[0]) - norm.cdf((cL1-mu[0])/s[0])
                GaussCDF2 = norm.cdf((cU2-mu[1])/s[1]) - norm.cdf((cL2-mu[1])/s[1])
                res += max(0, Psi1*Psi2-sPlus*GaussCDF1*GaussCDF2)
        return res
