import os, sys
import argparse
import numpy as np
import pycutest
from utils import load_results

HERE_TEST_PATH = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(HERE_TEST_PATH, './genosolver/')):
    sys.path.insert(0, os.path.join(HERE_TEST_PATH, './genosolver/'))
    import genosolver
    sys.path.pop(0)
else:
    raise ImportError('No genosolver folder')

if os.path.exists(os.path.join(HERE_TEST_PATH, "./bayesian-geno/")):
    sys.path.insert(0, os.path.join(HERE_TEST_PATH, "./bayesian-geno/"))

    import bayesian_line_search.GPgenosolver

    sys.path.pop(0)
else:
    print(HERE_TEST_PATH)
    raise ImportError("No bayesian genosolver folder")

def check_return_values(solver_name, solver_result):
    if solver_result['fun'] != solver_result['fun2']:
        print(f"{solver_name} has different fun={solver_result['fun']} and fun2={solver_result['fun2']} delta={solver_result['fun'] - solver_result['fun2']}")

def check_bound_violations(solver_name, solver_result):
    violations = np.sum(np.array(solver_result['x_bounds_violated']) != 0)
    if violations > 0:
        print(f"{solver_name} has invalid x at {violations} function evaluations")

def check_problem_properties(problem):
    if (problem.x0 < problem.bl).any() or (problem.bu < problem.x0).any():
        print(f"x0 violates bounds")

def import_problem(problem, problem_name):
    if problem is None:
        problem = pycutest.import_problem(problem_name)
        check_problem_properties(problem)
    return problem

def check_result_bound(problem, solver_name, solver_result):
    if not ((problem.bl <= solver_result['x']).all() and (solver_result['x'] <= problem.bu).all()):
        print(f"{solver_name} result violates bound constraints")
    if solver_result['fun'] != problem.obj(solver_result['x']):
        print(f"{solver_name} result f {solver_result['fun']} does not match x f {problem.obj(solver_result['x'])}")

def get_f_best(problem_properties, solver_result):
    if problem_properties['constraints'] == "unconstrained":
        return np.nanmin(solver_result["fs"])
    
    valid_idx = np.where(np.array(solver_result['x_bounds_violated']) == 0)[0]
    if len(valid_idx) == 0:
        return None
    return solver_result['fs'][valid_idx[np.nanargmin(np.array(solver_result['fs'])[valid_idx])]]

def check_return_values_sampled(solver_name, solver_result, f_best):
    if f_best < solver_result['fun']:
        print(f"{solver_name} did not return best function value sampled. Best={f_best} and fun={solver_result['fun']}")
    if f_best < solver_result['fun2']:
        print(f"{solver_name} did not return best x sampled. Best={f_best} and fun2={solver_result['fun2']}")

    if f_best > solver_result['fun']:
        print(f"{solver_name} has better function value than best sampled. Best={f_best} and fun={solver_result['fun']}")
    if f_best > solver_result['fun2']:
        print(f"{solver_name} has better x than best sampled. Best={f_best} and fun2={solver_result['fun2']}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pickle helper.', add_help=False)
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='For loading .pkl results of a run.')
    parser.add_argument('-f', '--file', nargs='+', type=str, help='Results to load.')
    args = parser.parse_args()

    all_results = [load_results(f) for f in args.file]

    for result in all_results:
        for problem_name, problem_results in result.items():
            print(f"-------------------{problem_name}-------------------")
            problem = None
            problem_properties = pycutest.problem_properties(problem_name)
            for solver_name, solver_result in problem_results.items():
                check_return_values(solver_name, solver_result)

                # if len(solver_result["fs"]) == 0:
                #     print(f"{solver_name} has no function values")
                #     continue

                f_best = get_f_best(problem_properties, solver_result)

                check_return_values_sampled(solver_name, solver_result, f_best)

                check_bound_violations(solver_name, solver_result)

                if not any(solver_result['x_bounds_violated']): continue

                problem = import_problem(problem, problem_name)
                check_result_bound(problem, solver_name, solver_result)
