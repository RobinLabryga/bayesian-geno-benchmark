from scipy.optimize import minimize
from scipy.optimize._optimize import MemoizeJac
import time
import numpy as np

from .BaseSolver import BaseSolver


class ScipySolver(BaseSolver):
    """Scipy wrapper.
    """

    def __init__(self, optimization_problem, options=None, timeout=float('inf')):
        super().__init__(optimization_problem, options, timeout)

        self.f = MemoizeJac(self.fg_with_timing)
        self.g = self.f.derivative


    def solve(self):
        self.reset()

        x = self.optimization_problem.x0
        lb = self.optimization_problem.bl
        ub = self.optimization_problem.bu
        lb[lb == -1e20] = -np.inf
        ub[ub ==  1e20] =  np.inf
        bounds = list(zip(lb, ub))

        # TODO This doesn't work for some reason, freezes on some problems, fix sometime later?
        if self.optimization_problem.cl is None and self.optimization_problem.cu is None:
            constraints = ()
        else:
            constraints = []
            eq_idxs = self.optimization_problem.is_eq_cons.nonzero()[0]
            ineq_idxs = (~self.optimization_problem.is_eq_cons).nonzero()[0]
            if sum(self.optimization_problem.is_eq_cons) > 0:
                cons = dict()
                cons['type'] = 'eq'
                def consjac(x):
                    ret = [self.optimization_problem.cons(x, index=i, gradient=True) for i in eq_idxs]
                    return np.array(list(map(lambda x: x[0], ret))), np.stack(list(map(lambda x: x[1], ret)))
                cons['fun'] = MemoizeJac(consjac)
                cons['jac'] = cons['fun'].derivative
                constraints.append(cons)
            if sum(~self.optimization_problem.is_eq_cons) > 0:
                cons = dict()
                cons['type'] = 'ineq'
                ineq_signs = [1 if self.optimization_problem.cl[i] == 0 else -1 for i in ineq_idxs]
                def consjac(x):
                    ret = [tuple(ineq_signs[j] * c_or_j for c_or_j in self.optimization_problem.cons(x, index=i, gradient=True)) for j, i in enumerate(ineq_idxs)]
                    return np.array(list(map(lambda x: x[0], ret))), np.stack(list(map(lambda x: x[1], ret)))
                cons['fun'] = MemoizeJac(consjac)
                cons['jac'] = cons['fun'].derivative
                constraints.append(cons)

        self.start_time = time.perf_counter()
        res = minimize(self.f, x, jac=self.g, bounds=bounds, constraints=constraints, options=self.options)
        self.end_time = time.perf_counter()
        self.result = res
        self.result.time = (self.end_time - self.start_time)
        self.result.fs = self.fs
        self.result.ts = self.ts
        self.result.pgs = self.pgs

    def getResult(self):
        return self.result
