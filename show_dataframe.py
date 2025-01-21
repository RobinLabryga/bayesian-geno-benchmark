import os, sys
import argparse
import numpy as np
import pandas as pd
import pycutest
from collections import defaultdict
from utils import load_results, save_results

HERE_TEST_PATH = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(HERE_TEST_PATH, "./genosolver/")):
    sys.path.insert(0, os.path.join(HERE_TEST_PATH, "./genosolver/"))
    import genosolver

    sys.path.pop(0)
else:
    raise ImportError("No genosolver folder")

if os.path.exists(os.path.join(HERE_TEST_PATH, "./bayesian-geno/")):
    sys.path.insert(0, os.path.join(HERE_TEST_PATH, "./bayesian-geno/"))

    import bayesian_line_search.GPgenosolver

    sys.path.pop(0)
else:
    print(HERE_TEST_PATH)
    raise ImportError("No bayesian genosolver folder")

TARGET_SOLVER = "BayesianGenoSolver"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pickle helper.", add_help=False)
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="For loading .pkl results of a run.",
    )
    parser.add_argument("-f", "--file", nargs="+", type=str, help="Results to load.")
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
    parser.add_argument(
        "--plot_again", action="store_true", help="Plot again like main.py does"
    )
    args = parser.parse_args()

    all_results = [load_results(f) for f in args.file]

    keys = [
        "x",
        "x_new",
        "fun",
        "fun2",
        "nit",
        "nfev",
        "status",
        "success",
        "message",
        "time",
        "pgnorm",
        "fs",
        "bounds_violated",
        "pgs",
    ]
    Dc = defaultdict(list)
    for results in all_results:
        for task, res in results.items():
            for nem, dc in res.items():
                Dc["solver"].append(nem)
                Dc["task"].append(task)
                for key in keys:
                    Dc[key].append(dc[key] if key in dc else -1)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    dframe = pd.DataFrame.from_dict(Dc).sort_values(["task", "solver"])
    opt = dframe.groupby("task")["fun"].transform("min")
    dframe["opt"] = opt
    dframe["opt2"] = dframe.groupby("task")["fun2"].transform("min")
    dframe["best"] = dframe["fun2"] == dframe["opt2"]
    dframe["(f-opt)/(abs(opt)+1)"] = dframe.groupby("task")["fun2"].transform(
        lambda x: (x - x.min()) / (abs(x.min()) + 1)
    )
    dframe["rel_pgnorm"] = dframe["pgnorm"] / (abs(dframe["fun2"]) + 1)
    # print(dframe)

    print("----------function value solved----------")

    print(
        dframe.groupby("solver").agg(
            {"(f-opt)/(abs(opt)+1)": (lambda x: sum(x < 1e-4))}
        )
    )

    print("\nProblems that were not solved")
    print(
        dframe[dframe["(f-opt)/(abs(opt)+1)"] >= 1e-4]
        .groupby("solver")["task"]
        .apply(list)
    )

    print(f"(f-opt)/(abs(opt)+1) >= 1e-4: compared to best:")
    bad_f = dframe[
        (dframe["solver"] == TARGET_SOLVER) & (dframe["(f-opt)/(abs(opt)+1)"] >= 1e-4)
    ]
    extended_bad_f = dframe[
        dframe["task"].isin(bad_f["task"])
        & (dframe["best"] | (dframe["solver"] == TARGET_SOLVER))
    ]
    print(
        extended_bad_f[
            [
                "task",
                "solver",
                "message",
                "fun",
                "fun2",
                "rel_pgnorm",
                "(f-opt)/(abs(opt)+1)",
            ]
        ]
    )

    print("----------gradient solved----------")

    print(dframe.groupby("solver").agg({"rel_pgnorm": (lambda x: sum(x < 1e-6))}))

    print("\nProblems that were not solved")
    print(dframe[dframe["rel_pgnorm"] >= 1e-6].groupby("solver")["task"].apply(list))

    print("rel_pgnorm >= 1e-6 compared to best")
    pg_bad = dframe[
        (dframe["solver"] == TARGET_SOLVER) & (dframe["rel_pgnorm"] >= 1e-6)
    ]
    extended_pg_bad = dframe[
        dframe["task"].isin(pg_bad["task"])
        & (dframe["best"] | (dframe["solver"] == TARGET_SOLVER))
    ]
    print(
        extended_pg_bad[
            [
                "task",
                "solver",
                "message",
                "fun",
                "fun2",
                "rel_pgnorm",
                "(f-opt)/(abs(opt)+1)",
            ]
        ]
    )

    print("----------Converge in f and/or g----------")

    print("f or g")
    dframe["f_or_g_opt"] = (dframe["(f-opt)/(abs(opt)+1)"] < 1e-4) | (
        dframe["rel_pgnorm"] < 1e-6
    )
    print(dframe.groupby("solver").agg({"f_or_g_opt": (lambda x: sum(1 * x))}))

    print(f"not f or g compared to best:")
    bad_f_or_g = dframe[
        (dframe["solver"] == TARGET_SOLVER) & (dframe["f_or_g_opt"] == False)
    ]
    extended_bad_f_or_g = dframe[
        dframe["task"].isin(bad_f_or_g["task"])
        & (dframe["best"] | (dframe["solver"] == TARGET_SOLVER))
    ]
    print(
        extended_bad_f_or_g[
            [
                "task",
                "solver",
                "message",
                "fun",
                "fun2",
                "rel_pgnorm",
                "(f-opt)/(abs(opt)+1)",
            ]
        ]
    )

    print("f and g")
    dframe["f_and_g_opt"] = (dframe["(f-opt)/(abs(opt)+1)"] < 1e-4) & (
        dframe["rel_pgnorm"] < 1e-6
    )
    print(dframe.groupby("solver").agg({"f_and_g_opt": (lambda x: sum(1 * x))}))

    print("not f and g")
    dframe["not_f_g_opt"] = (dframe["(f-opt)/(abs(opt)+1)"] >= 1e-4) & (
        dframe["rel_pgnorm"] < 1e-6
    )
    print(dframe.groupby("solver").agg({"not_f_g_opt": (lambda x: sum(1 * x))}))

    print("f and not g")
    dframe["f_not_g_opt"] = (dframe["(f-opt)/(abs(opt)+1)"] < 1e-4) & (
        dframe["rel_pgnorm"] >= 1e-6
    )
    print(dframe.groupby("solver").agg({"f_not_g_opt": (lambda x: sum(1 * x))}))

    print("not f and not g")
    dframe["not_f_not_g_opt"] = (dframe["(f-opt)/(abs(opt)+1)"] >= 1e-4) & (
        dframe["rel_pgnorm"] >= 1e-6
    )
    print(dframe.groupby("solver").agg({"not_f_not_g_opt": (lambda x: sum(1 * x))}))
    print(dframe[dframe["not_f_not_g_opt"]].groupby("solver")["task"].apply(list))

    if args.plot_again:
        results = defaultdict(dict)
        for result in all_results:
            for task, res in result.items():
                for nem, dc in res.items():
                    results[task][nem] = dc
        metrics = ["nit", "nfev", "time"]
        save_results(
            results,
            metrics,
            {},
            {},
            one_fig_pdf=(not args.single_fig_per_f_over_time),
            barplots_cnt=args.barplots_cnt,
        )
