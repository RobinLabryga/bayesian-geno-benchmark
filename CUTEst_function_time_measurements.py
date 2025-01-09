import pycutest
import numpy as np
from time import time
import json
import pycutest

if __name__ == "__main__":
    constraint = "bound"
    problemNames = pycutest.find_problems(constraints=constraint)
    print(f"There are {len(problemNames)} {constraint} problems")

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
            x = x + rng.uniform() * g

        measurements[problemName] = {
            "n_samples": n_samples,
            "total": sum(samples),
            "median": np.median(samples),
            "mean": np.mean(samples),
            "var": np.var(samples),
            "std": np.std(samples),
            "min": min(samples),
            "max": max(samples),
        }

    for problem, measurement in measurements.items():
        print(
            f"{problem}: median={measurement['median']}, mean={measurement['mean']}, var={measurement['var']}, std={measurement['std']}"
        )

    with open(f"problem_time_measurements_{constraint}.json", "w") as fp:
        json.dump(measurements, fp, sort_keys=True, indent=4)
