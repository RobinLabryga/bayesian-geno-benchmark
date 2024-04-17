import autograd.numpy as np

class Modelquadratic:

    def __init__(self):
        self.N = 1000
        self.A = np.ones((self.N, self.N))
        # Variables
        self.x = 2 * np.ones(self.N)
        self.x_lower = np.ones(self.N) * -np.inf
        self.x_upper = np.ones(self.N) * np.inf

    def _mergeVars(self, x):
        return x
    
    def _unmergeVars(self, _vars):
        return _vars

    def getStartingPoint(self):
        return self._mergeVars(self.x)

    def getLowerBounds(self):
        return self._mergeVars(self.x_lower)

    def getUpperBounds(self):
        return self._mergeVars(self.x_upper)

    def _f(self, _vars):
        x = self._unmergeVars(_vars)
        return np.dot(x, self.A @ x)

    def g(self, _vars):
        x = self._unmergeVars(_vars)
        return self.A @ x + self.A.T @ x

    def H(self, _vars):
        x = self._unmergeVars(_vars)
        return self.A + self.A.T


opt = Modelquadratic()
