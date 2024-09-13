import argparse
import json
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare function evaluation and other computation.", add_help=False
    )
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="For comparing function evaluation and other computation.",
    )
    parser.add_argument(
        "--functionfile",
        "-f",
        type=str,
        default="./time_results/problem_time_measurements.json",
        help="Paths to function file.",
    )
    parser.add_argument(
        "--solverfiles",
        "-s",
        type=str,
        nargs="+",
        required=True,
        help="The json file of the solver times and fevals.",
    )
    args = parser.parse_args()

    # Change config to 'all' to run all solvers on all problems

    with open(args.functionfile, "r") as f:
        function_times = json.load(f)

    for solverfile in args.solverfiles:
        print(f"{solverfile}:")

        with open(solverfile, "r") as f:
            solver_times = json.load(f)

        times_per_feval = list()

        for problem_name in solver_times.keys():
            function_info = function_times[problem_name]
            solver_info = solver_times[problem_name]
            # fun = solver_info['nfev'] * function_info['mean']
            # other = solver_info['time'] - fun
            # ratio = fun / solver_info['time']
            # comp_per_feval = other / solver_info['nfev']
            # print(f"{problem_name} fun: {fun}, other:{other}, ratio:{ratio}, comp/feval:{comp_per_feval}")
            times_per_feval.append(solver_info["time"] / solver_info["nfev"] - function_info['mean'])

        print(
            f"mean: {np.mean(times_per_feval)}, std: {np.std(times_per_feval)}, min: {min(times_per_feval)}, max: {max(times_per_feval)}"
        )
        print(
            f"{np.mean(times_per_feval) * 1000:.2f} & {np.std(times_per_feval) * 1000:.2f} & {max(times_per_feval) * 1000:.2f}"
        )
