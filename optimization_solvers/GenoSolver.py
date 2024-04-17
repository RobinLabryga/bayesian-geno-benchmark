# use local geno
import sys
import os
import time
import numpy as np

from .BaseSolver import BaseSolver

# TODO use pip geno later
HERE_TEST_PATH = os.path.dirname(os.path.abspath(__file__))

if os.path.exists(os.path.join(HERE_TEST_PATH, '../genosolver')):
    sys.path.insert(0, os.path.join(HERE_TEST_PATH, '../genosolver/'))
    from genosolver.genosolver import minimize
    sys.path.pop(0)
else:
    raise ImportError('No genosolver folder')


class GenoSolver(BaseSolver):
    """Genosolver wrapper.
    """

    def __init__(self, optimization_problem, options=None, timeout=float('inf')):
        super().__init__(optimization_problem, options, timeout)

    def solve(self):
        self.reset()

        x = self.optimization_problem.x0
        lb = self.optimization_problem.bl
        ub = self.optimization_problem.bu
        lb[lb == -1e20] = -np.inf
        ub[ub ==  1e20] = np.inf

        if self.optimization_problem.cl is None and self.optimization_problem.cu is None:
            constraints = None
        else:
            constraints = dict()
            constraints['fun'] = self.optimization_problem.cons
            constraints['lb'] = self.optimization_problem.cl
            constraints['ub'] = self.optimization_problem.cu
            constraints['jacprod'] = lambda x, v: self.optimization_problem.jprod(v, transpose=True, x=x)

        self.start_time = time.perf_counter()
        res = minimize(self.fg_with_timing, x, lb, ub, self.options, constraints)
        self.end_time = time.perf_counter()
        self.result = res
        self.result.time = (self.end_time - self.start_time)
        self.result.fs = self.fs
        self.result.ts = self.ts
        self.result.pgs = self.pgs


    def getResult(self):
        return self.result
