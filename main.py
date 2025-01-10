import os, sys, gc
import argparse
import cProfile
import pycutest
from tqdm import tqdm
import numpy as np

from utils import (
    save_results,
    setup_from_config_file,
    create_failed_result,
    create_additional_info,
)


if __name__ == "__main__":
    available_problems = pycutest.find_problems(constraints="unconstrained")
    available_solvers = list(
        map(
            lambda f: f[:-3],
            filter(
                lambda f: f != "__init__.py" and f.endswith(".py"),
                os.listdir("optimization_solvers"),
            ),
        )
    )

    available_problems = sorted(available_problems)
    available_solvers = sorted(available_solvers)

    parser = argparse.ArgumentParser(description="Benchmark solvers.", add_help=False)
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help=f"Available problems are: {available_problems}. Available solvers are {available_solvers}",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./example_config.yml",
        help="Path to .yaml config file to use for benchmarks. By default './example_config.yml'",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not save to results/ directory."
    )
    parser.add_argument(
        "--single_fig_per_f_over_time",
        action="store_true",
        help="Save one f_over_time plot per file.",
    )
    parser.add_argument(
        "--barplots_cnt",
        type=int,
        default=-1,
        help="Number of barplots per pdf. -1 means all.",
    )
    parser.add_argument("--profile", action="store_true", help="run profiler on solve")
    parser.add_argument(
        "--timeout",
        type=float,
        default=float("inf"),
        help="Time after which the test will timeout. By default inf or solver defined.",
    )
    parser.add_argument(
        "--result_dir", type=str, help="Name of the directory inside results/"
    )
    parser.add_argument("--print_results", action="store_true", help="TBA")
    args = parser.parse_args()

    config = args.config
    problem_dict, solver_dict, problems, solvers = setup_from_config_file(
        config, available_problems, available_solvers
    )

    print(f"Solving {len(problems)} problems...")
    results = dict()
    for problem_name in tqdm(problems):
        results[problem_name] = dict()
        for solver_name in solver_dict.keys():
            try:
                problem = pycutest.import_problem(
                    problem_name, sifParams=problem_dict[problem_name]
                )
            except RuntimeError:
                print(f"Failed to load {problem_name}")
                if problem_name in results:
                    results.pop(problem_name)
                break

            solver = solvers[solver_name]
            s = solver(problem, solver_dict[solver_name]["options"], args.timeout)
            try:
                tqdm.write(f"Starting {problem.name} with {solver.__name__}")
                if args.profile:
                    cProfile.runctx("s.solve()", globals(), locals())
                else:
                    s.solve()
                tqdm.write(
                    f"Finished {problem.name} with {solver.__name__} in {s.getResult().time} seconds"
                )
                res = s.getResult()
            except TimeoutError:
                tqdm.write(f"Timed out on {problem_name} with {solver_name}")
                res = create_failed_result(
                    problem,
                    status=101,
                    message="Timed out",
                    xs=s.xs,
                    fs=s.fs,
                    ts=s.ts,
                    pgs=s.pgs,
                )
            except KeyboardInterrupt:
                tqdm.write(f"Interrupted {problem_name} with {solver_name}")
                res = create_failed_result(
                    problem,
                    status=102,
                    message="Keyboard interrupt",
                    xs=s.xs,
                    fs=s.fs,
                    ts=s.ts,
                    pgs=s.pgs,
                )
            except BaseException as error:
                tqdm.write(
                    f"Broke on {problem_name} with {solver_name}\nError message: {error}"
                )
                res = create_failed_result(
                    problem,
                    status=103,
                    message=str(error),
                    xs=s.xs,
                    fs=s.fs,
                    ts=s.ts,
                    pgs=s.pgs,
                )
            res_add = create_additional_info(problem, res["x"])
            res.update(res_add)

            res['x_valid'] = [(problem.bl <= x).all() and (x <= problem.bu).all() for x in s.xs]

            results[problem_name][solver_name] = res
            if args.print_results:
                print(res)

    if not args.no_save:
        gc.collect()
        metrics = ["nit", "nfev", "time"]
        save_results(
            results,
            metrics,
            solver_dict,
            problem_dict,
            one_fig_pdf=(not args.single_fig_per_f_over_time),
            barplots_cnt=args.barplots_cnt,
            result_dir=args.result_dir,
        )
