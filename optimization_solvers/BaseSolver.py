import time
import numpy as np

class BaseSolver:
    """Base class for optimization solvers. Loads problem, takes care of feval...
    """

    def __init__(self, optimization_problem, options=None, timeout=float('inf')):
        self.optimization_problem = optimization_problem
        self.options = options
        self.timeout = timeout
        self.reset()

    def reset(self):
        self.xs = []  # memory goes brrrr
        self.fs = []
        self.pgs = []
        self.ts = []
        self.start_time = None  # need to init later
        self.end_time = None
        self.result = None

    def fg_with_timing(self, x):
        f, g = self.optimization_problem.obj(x, gradient=True)
        lb = self.optimization_problem.bl
        ub = self.optimization_problem.bu
        lb[lb == -1e20] = -np.inf
        ub[ub ==  1e20] = np.inf

        eps = 1E-10
        working = np.full(len(x), 1.0)
        working[(x <= lb + eps * 2) & (g >= 0)] = 0
        working[(x >= ub - eps * 2) & (g <= 0)] = 0
        pg = np.linalg.norm(g[working > 0], np.inf) if any(working > 0) else 0.

        self.xs.append(x)
        self.fs.append(f)
        self.pgs.append(pg)
        self.ts.append(time.perf_counter() - self.start_time)
        #self.xs = self.xs[-1000:] ## Cuz of RAM
        #self.fs = self.fs[-1000:] ## ~||~
        #self.ts = self.ts[-1000:] ## ~||~
        if self.ts[-1] > self.timeout:
            raise TimeoutError(f'Solver timed out on {self.optimization_problem.name}')
        return f, g


    def solve(self):
        # will want to warm up with a few funcalls to fg_with_timing?
        pass
