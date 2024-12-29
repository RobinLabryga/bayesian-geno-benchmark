import os
import re
import datetime
import pickle
import yaml
import csv
import json
import importlib
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pycutest

from scipy.optimize import OptimizeResult


def parse_option(s: str):
    """Turn s into an integer if possible. Used in arg parsing."""
    if (s[0] in ("-", "+") and s[1:].isdigit()) or s.isdigit():
        return int(s)
    return s


def setup_from_config_file(handle, available_problems, available_solvers):
    """Return problems and solvers contained in handle, with their respective options."""
    assert os.path.exists(handle), f"Config file {handle} doesn't exist"
    with open(handle, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            return e

    if config["problems"] == "all":
        problems = pycutest.find_problems()
    elif "single_problems" not in config.keys():
        # instead of single problems, we're loading them by category now
        # see https://jfowkes.github.io/pycutest/_build/html/functions/pycutest.find_problems.html
        # TODO add sifParams to config yaml
        problems = pycutest.find_problems(**config["problems"])
    else:
        # load single problems
        problems = config["single_problems"].keys()

    problem_dict = dict()
    for problem in problems:
        problem_dict[problem] = None

    # get solvers the same as before
    solver_dict = config["solvers"]
    for solver_name in solver_dict.keys():
        if solver_dict[solver_name] is None:
            solver_dict[solver_name] = {"options": {}}

    solvers = find_solvers(solver_dict.keys())

    return problem_dict, solver_dict, problems, solvers


def decode_dict(d: dict):
    """Convert bytes to str for yaml dumping."""
    res = {}
    for key, value in d.items():
        if isinstance(key, bytes):
            key = key.decode()
        if isinstance(value, bytes):
            value = value.decode()
        elif isinstance(value, dict):
            value = decode_dict(value)
        res[key] = value
    return res


def save_results(
    results: dict,
    metrics: list,
    solver_dict: dict,
    problem_dict: dict,
    one_fig_pdf=True,
    barplots_cnt=-1,
    result_dir=None,
):
    # result dir as date
    if result_dir is None:
        result_dir = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    if not os.path.exists(f"./results/{result_dir}"):
        os.makedirs(f"./results/{result_dir}")
    with open(f"./results/{result_dir}/results.pkl", "wb") as f:
        pickle.dump(results, f)

    plot_metrics_barplot(
        results, metrics, f"./results/{result_dir}", barplots_cnt=barplots_cnt
    )
    plot_f_over_time(results, f"./results/{result_dir}", one_fig_pdf=one_fig_pdf)
    plot_f_over_feval(results, f"./results/{result_dir}", one_fig_pdf=one_fig_pdf)
    plot_normalized_feval(results, f"./results/{result_dir}")
    plot_solved_over_time(results, f"./results/{result_dir}")
    plot_metrics_pdf(results, metrics, f"./results/{result_dir}")
    solver_times_to_json(results, f"./results/{result_dir}")

    result_dict = dict()
    result_dict["solvers"] = solver_dict
    result_dict["problems"] = problem_dict
    with open(f"./results/{result_dir}/config.yml", "w") as f:
        f.write(yaml.dump(decode_dict(result_dict)))


def load_results(results_handle: str):
    assert os.path.exists(results_handle), f"Result file {results_handle} doesn't exist"
    with open(results_handle, "rb") as f:
        results = pickle.load(f)
    return results


def find_problems(problem_names: list):
    """Returns some optimization problem classes from './optimization_problems.
    Unused while using pycutest.
    """
    available_problems = list(
        map(
            lambda f: f[:-3],
            filter(
                lambda f: f != "__init__.py" and f.endswith(".py"),
                os.listdir("optimization_problems"),
            ),
        )
    )
    problems = dict()
    for problem_name in problem_names:
        assert (
            problem_name in available_problems
        ), f"Problem {problem_name} doesn't exist"
        module = importlib.import_module(f"optimization_problems.{problem_name}")
        class_name = next(
            filter(
                lambda attr: re.search(problem_name, attr, re.IGNORECASE), dir(module)
            )
        )
        problems[problem_name] = getattr(module, class_name)
    return problems


def find_solvers(solver_names: list):
    """Returns some solver classes from './solvers."""
    available_solvers = list(
        map(
            lambda f: f[:-3],
            filter(
                lambda f: f != "__init__.py" and f.endswith(".py"),
                os.listdir("optimization_solvers"),
            ),
        )
    )
    solvers = dict()
    for solver_name in solver_names:
        assert solver_name in available_solvers, f"Solver {solver_name} doesn't exist"
        module = importlib.import_module(f"optimization_solvers.{solver_name}")
        class_name = next(
            filter(
                lambda attr: re.search(solver_name, attr, re.IGNORECASE), dir(module)
            )
        )
        solvers[solver_name] = getattr(module, class_name)
    return solvers


def plot_metrics_barplot(
    results: dict, metrics: List[str], result_dir: str, barplots_cnt=-1
):
    if barplots_cnt == -1:
        barplots_cnt = len(results)
    results_list = list(results.items())
    for batch in range(0, len(results), barplots_cnt):
        results = dict(results_list[batch : batch + barplots_cnt])
        num_problems = len(results)
        num_solvers = len(results[next(iter(results))])  # ugly
        num_metrics = len(metrics)
        # plot all metrics into one file
        fig, axs = plt.subplots(
            num_problems,
            num_metrics,
            figsize=(3 * num_metrics * num_solvers, 4 * num_problems),
            squeeze=False,
        )
        for i, (problem_name, solver_result) in enumerate(results.items()):
            for j, met in enumerate(metrics):
                axs[i][j].set_title(problem_name)
                axs[i][j].set_ylabel(met)
                curr_barplot = axs[i][j].bar(
                    solver_result.keys(),
                    [r[met] for r in solver_result.values()],
                    label=met,
                )
                for solver_name, curr_bar in zip(solver_result.keys(), curr_barplot):
                    if solver_result[solver_name].status == 0:
                        axs[i][j].text(
                            curr_bar.get_x() + curr_bar.get_width() / 2.0,
                            curr_bar.get_height(),
                            f"{curr_bar.get_height()}",
                            ha="center",
                            va="bottom",
                        )
                    else:
                        axs[i][j].text(
                            curr_bar.get_x() + curr_bar.get_width() / 2.0,
                            curr_bar.get_height(),
                            solver_result[solver_name].message[:15],
                            ha="center",
                            va="bottom",
                        )

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        fig.savefig(f"{result_dir}/barplot_{batch}.pdf", format="pdf")
        plt.close()


def plot_f_over_time(results: dict, result_dir: str, one_fig_pdf=True):
    num_problems = len(results)
    if one_fig_pdf:
        fig, axs = plt.subplots(
            num_problems, figsize=(20, 4 * num_problems), squeeze=False
        )
    else:
        axs = [None] * num_problems
    for (problem_name, solver_result), ax in zip(results.items(), axs):
        if not one_fig_pdf:
            fig, axs = plt.subplots(1, figsize=(20, 4), squeeze=False)
            ax = axs[0]

        ax[0].set_title(problem_name)
        ax[0].set_xlabel("time (s)")
        ax[0].set_ylabel("f val")
        ax[0].set_yscale("symlog")
        legend = []
        for s, r in solver_result.items():
            if len(r["fs"]) == 0:
                continue
            legend.append(s)
            idxs, mn = [0], r["fs"][0]
            for j in range(1, len(r["fs"])):
                if r["fs"][j] <= mn:
                    idxs.append(j)
                mn = min(mn, r["fs"][j])
            ax[0].plot(
                np.array(r["ts"])[idxs],
                np.array(r["fs"])[idxs],
                linewidth=0.5,
                marker=".",
            )
        ax[0].legend(legend)
        if not one_fig_pdf:
            if not os.path.exists(f"{result_dir}/{problem_name}"):
                os.makedirs(f"{result_dir}/{problem_name}")
            fig.savefig(f"{result_dir}/{problem_name}/f_over_time.pdf", format="pdf")
            plt.close()
    if one_fig_pdf:
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        fig.savefig(f"{result_dir}/f_over_time.pdf", format="pdf")
        plt.close()


def plot_f_over_feval(results: dict, result_dir: str, one_fig_pdf=True):
    num_problems = len(results)
    if one_fig_pdf:
        fig, axs = plt.subplots(
            num_problems, figsize=(20, 4 * num_problems), squeeze=False
        )
    else:
        axs = [None] * num_problems
    for (problem_name, solver_result), ax in zip(results.items(), axs):
        if not one_fig_pdf:
            fig, axs = plt.subplots(1, figsize=(20, 4), squeeze=False)
            ax = axs[0]

        ax[0].set_title(problem_name)
        ax[0].set_xlabel("f eval")
        ax[0].set_ylabel("f val")
        ax[0].set_yscale("symlog")
        legend = []
        for s, r in solver_result.items():
            if len(r["fs"]) == 0:
                continue
            legend.append(s)
            idxs, mn = [0], r["fs"][0]
            for j in range(1, len(r["fs"])):
                if r["fs"][j] < mn:
                    idxs.append(j)
                mn = min(mn, r["fs"][j])
            ax[0].plot(
                np.array(idxs) + 1, np.array(r["fs"])[idxs], linewidth=0.5, marker="."
            )
        ax[0].legend(legend)
        if not one_fig_pdf:
            if not os.path.exists(f"{result_dir}/{problem_name}"):
                os.makedirs(f"{result_dir}/{problem_name}")
            fig.savefig(f"{result_dir}/{problem_name}/f_over_feval.pdf", format="pdf")
            plt.close()
    if one_fig_pdf:
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        fig.savefig(f"{result_dir}/f_over_feval.pdf", format="pdf")
        plt.close()


def solver_times_to_json(results: dict, result_dir: str):
    output = dict()
    for problem_name, problems_results in results.items():
        for solver_name, solver_result in problems_results.items():
            if solver_name not in output:
                output[solver_name] = dict()
            output[solver_name][problem_name] = dict()
            output[solver_name][problem_name]["time"] = solver_result["time"]
            output[solver_name][problem_name]["nfev"] = len(solver_result["fs"])

    for solver_name, out in output.items():
        with open(f"{result_dir}/{solver_name}_times.json", "w") as f:
            json.dump(out, f, sort_keys=True, indent=4)


def plot_normalized_feval(results: dict, result_dir: str):
    fig, ax = plt.subplots(1, 1)
    ax.set_title("f evals")
    ax.set_xlabel("norm f eval")
    ax.set_ylabel("norm f val")
    # ax.set_xscale("symlog")
    # ax.set_yscale("symlog")

    solver_lines = dict()

    for problem_name, problems_results in results.items():
        feval_min = None
        f_best = None
        for solver_name, solver_result in problems_results.items():
            if len(solver_result["fs"]) == 0:
                continue
            min_index = np.nanargmin(solver_result["fs"])
            min_value = solver_result["fs"][min_index]
            if f_best is None or min_value < f_best:
                feval_min = min_index + 1
                f_best = min_value
            elif min_value == f_best:
                feval_min = min(feval_min, min_index + 1)

        if feval_min is None:
            continue

        if f_best == -np.inf:
            continue

        for solver_name, solver_result in problems_results.items():
            if len(solver_result["fs"]) == 0:
                continue

            if solver_name not in solver_lines:
                solver_lines[solver_name] = dict()
            solver_lines[solver_name][problem_name] = dict()
            fs = solver_result["fs"]

            best_fs = list()
            best_f = fs[0]
            for f in fs:
                best_f = min(best_f, f)
                best_fs.append(best_f)

            f_max = best_fs[0]

            solver_lines[solver_name][problem_name]["normalized_fs"] = (
                [(f - f_best) / (f_max - f_best) for f in best_fs]
                if f_max - f_best != 0
                else [1.0] * len(best_fs)
            )
            solver_lines[solver_name][problem_name]["normalized_fevals"] = [
                feval / feval_min for feval in range(len(fs))
            ]

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for solver_name, solver_line in solver_lines.items():
        f_evals = np.linspace(0, 4, 400)
        fs = [list() for _ in f_evals]
        for problem_name, problem_line in solver_line.items():
            normalized_fs = problem_line["normalized_fs"]
            normalized_fevals = np.array(problem_line["normalized_fevals"])
            for i in range(len(f_evals)):
                f_eval_index = normalized_fevals.searchsorted(f_evals[i], "right") - 1
                fs[i].append(normalized_fs[f_eval_index])

        f_min = [min(f) for f in fs]
        f_max = [max(f) for f in fs]
        f_mean = np.array([np.mean(f) for f in fs])
        f_std = np.array([np.std(f) for f, mean in zip(fs, f_mean)])
        std_lower = np.maximum(f_min, f_mean - f_std)
        std_upper = np.minimum(f_max, f_mean + f_std)

        ax.plot(f_evals, f_mean, label=solver_name)
        ax.fill_between(f_evals, std_lower, std_upper, alpha=0.25)

        with open(f"{result_dir}/feval_normalized{solver_name}.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(("feval", "mean", "std_lower", "std_upper"))
            for row in zip(f_evals, f_mean, std_lower, std_upper):
                writer.writerow(row)

    ax.legend()

    fig.savefig(f"{result_dir}/feval_normalized.pdf", format="pdf")
    plt.close()


def plot_solved_over_time(results: dict, result_dir: str, max_time=1000):
    solver_times = dict()
    plotted_time = 0
    for problem_name, solver_result in results.items():
        for solver_name, result in solver_result.items():
            if not solver_times.get(solver_name):
                solver_times[solver_name] = []
            solver_times[solver_name].append(
                result["time"] if result["success"] else 1e9
            )
            plotted_time = min(max_time, max(plotted_time, np.ceil(result["time"])))

    plt.figure(figsize=(20, 20))
    plt.ylim(0, 100)
    xx = np.linspace(0, plotted_time, max(1000, 2 * int(plotted_time)))
    for solver_name, times in solver_times.items():
        plt.plot(
            xx, 100 * (np.searchsorted(sorted(times), xx, side="right") / len(times))
        )
        plt.legend(solver_times.keys())
    plt.xlabel("time (s)")
    plt.ylabel("% solved")
    plt.savefig(f"{result_dir}/solved_over_time.pdf", format="pdf")
    plt.close()


def plot_metrics_pdf(
    results: dict, metrics: List[str], result_dir: str, success_only: bool = False
):
    assert success_only == False, "pdf for unsuccessful tries doesnt work yet"
    fig, axs = plt.subplots(
        len(metrics), 1, figsize=(20, 5 * len(metrics)), squeeze=False
    )
    problem_names = results.keys()
    solver_names = results[next(iter(problem_names))].keys()
    for i, met in enumerate(metrics):
        legend = []
        for s in solver_names:
            data = filter(
                lambda res: res["success"], [results[p][s] for p in problem_names]
            )
            data = list(map(lambda res: res[met], data))
            data = {met: data}
            sns.kdeplot(data=data, x=met, ax=axs[i, 0])
            legend.append(f"{s} (n = {len(data[met])})")
        axs[i, 0].legend(legend)
    plt.savefig(f"{result_dir}/pdf.pdf", format="pdf")


def create_failed_result(
    problem,
    status: int,
    message: str,
    xs: list = [],
    fs: list = [],
    ts: list = [],
    pgs: list = [],
):
    x = xs[fs.index(min(fs))] if len(fs) else problem.x0
    fun = min(fs) if len(fs) else 1e18
    res = OptimizeResult(
        x=x, success=False, status=status, fun=fun, nfev=len(fs), nit=0, message=message
    )
    res.fs = fs
    res.ts = ts
    res.time = ts[-1] if len(ts) else 0
    res.pgs = pgs
    res.gnorm = np.linalg.norm(res["jac"], np.inf) if "jac" in res.keys() else np.inf
    return res


def create_additional_info(problem, x: np.ndarray) -> dict:
    res = dict()
    lb = problem.bl
    ub = problem.bu
    lb[lb == -1e20] = -np.inf
    ub[ub == 1e20] = np.inf

    x_new = np.maximum(np.minimum(x, ub), lb)
    res["bounds_violated"] = np.sum(1 * (x_new != x))
    res["x_new"] = x_new

    f, g = problem.obj(x_new, gradient=True)
    res["fun2"] = f

    eps = 1e-10
    working = np.full(len(x), 1.0)
    working[(x <= lb + eps * 2) & (g >= 0)] = 0
    working[(x >= ub - eps * 2) & (g <= 0)] = 0
    pg = np.linalg.norm(g[working > 0], np.inf) if any(working > 0) else 0.0
    res["pgnorm"] = pg
    res["gnorm"] = np.linalg.norm(g, np.inf)
    return res
