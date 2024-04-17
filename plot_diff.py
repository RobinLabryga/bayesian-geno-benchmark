import os, sys, io
import argparse
import cProfile
import timeout_decorator
import numpy as np
from scipy.optimize import OptimizeResult
import matplotlib.pyplot as plt

from utils import load_results, plot_metrics_barplot, plot_f_over_time

from collections import defaultdict

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Benchmark solvers.', add_help=False)
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='For plotting .pkl results of runs.')
    parser.add_argument('--paths', '--pth', type=str, nargs='+', required=True, help='Paths to all results to be plotted.')
    parser.add_argument('--out', type=str, default='./diff_plots/', help='PDF directory where to save plotted results.')
    args = parser.parse_args()

    # Change config to 'all' to run all solvers on all problems
    
    all_results = list(map(load_results, args.paths))
    compact_results = defaultdict(dict)

    for (indx, results) in enumerate(all_results):
        for task in results:
            compact_results[task] |= { (solver + str(indx)) : res for (solver, res) in results[task].items() }

    for task in compact_results:
        for solver in compact_results[task]:
            compact_results[task][solver]['ts'] = list(range(len(compact_results[task][solver]['ts'])))
            
    os.makedirs(args.out, exist_ok=True)
    plot_metrics_barplot(compact_results, ['nfev', 'nit'], args.out)
    plot_f_over_time(compact_results, args.out)
    
