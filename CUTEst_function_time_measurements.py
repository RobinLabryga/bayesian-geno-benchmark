import pycutest
import numpy as np
from time import time
import json
import pycutest

if __name__ == '__main__':
    problemNames = pycutest.find_problems(constraints="unconstrained")
    print(f"There are {len(problemNames)} unconstrained problems")

    n_samples = 1_000

    measurements = dict()

    for problemName in problemNames:
        print(f"Measuring {problemName}")
        try:
            problem = pycutest.import_problem(problemName)
        except Exception as e:
            print(e)
            print(f"Failed to load {problemName}")
            continue
        samples = list()
        rng = np.random.default_rng()
        x = problem.x0
        for i in range(n_samples):
            start = time()
            f, g = problem.obj(problem.x0, gradient=True)
            samples.append(time() - start)
            x = x + rng.uniform(1.0, 8.0) * g

        mean = np.mean(samples)

        measurements[problemName] = {
            'n_samples': n_samples,
            'total': sum(samples),
            'median': np.median(samples),
            'mean': mean,
            'var': np.var(samples, mean=mean),
            'std': np.std(samples, mean=mean),
            'min': min(samples),
            'max': max(samples),
        }


    for problem, measurement in measurements.items():
        print(f"{problem}: median={measurement['median']}, mean={measurement['mean']}, var={measurement['var']}, std={measurement['std']}")

    with open('problem_time_measurements.json', 'w') as fp:
        json.dump(measurements, fp, sort_keys=True, indent=4)
