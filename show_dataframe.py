import os, sys, glob
import argparse
import numpy as np
import pandas as pd
import pycutest
from collections import defaultdict
from utils import load_results, save_results

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

TARGET_SOLVER = 'BayesianGenoSolver'
OTHER_SOLVER = 'ScipySolver'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pickle helper.', add_help=False)
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='For loading .pkl results of a run.')
    parser.add_argument('-f', '--file', nargs='+', type=str, default=max(glob.glob('./results/*/results.pkl'), key=os.path.getctime), help='Results to load.')
    parser.add_argument('--single_fig_per_f_over_time', action='store_true', help='Save one f_over_time plot per file.')
    parser.add_argument('--barplots_cnt', type=int, default=-1, help='Number of barplots per pdf. -1 means all.')
    parser.add_argument('--plot_again', action='store_true', help='Plot again like main.py does')
    args = parser.parse_args()

    all_results = [load_results(f) for f in args.file]

    keys = [ 'x', 'x_new', 'fun', 'fun2', 'nit', 'nfev', 'status', 'success', 'message', 'time', 'pgnorm', 'fs', 'bounds_violated', 'pgs' ]
    Dc = defaultdict(list)
    for results in all_results:
        for (task, res) in results.items():
            for (nem, dc) in res.items():
                Dc['solver'].append(nem)
                Dc['task'].append(task)
                for key in keys:
                    Dc[key].append(dc[key] if key in dc else -1)
    

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None) 
    pd.set_option('display.max_colwidth', None)

    dframe = pd.DataFrame.from_dict(Dc)
    opt = dframe.groupby('task')['fun'].transform('min')
    dframe['opt'] = opt
    dframe['opt2'] = dframe.groupby('task')['fun2'].transform('min')
    dframe['best'] = (dframe['fun2'] == dframe['opt2'])
    local_mins = dframe.groupby('task')['x'].transform(lambda x: np.linalg.norm([y for y in x] - np.array([ y for y in x])[:, None], ord=1, axis=-1).max())
    dframe['local_mins_dist'] = local_mins
    global_mins = dframe.groupby('task').apply(lambda x: list(np.linalg.norm(np.array([ y for y in x['x'] ])-x[x['best']]['x'].iloc[0], axis=1)))
    dframe['global_mins_dist'] = [ y for y in global_mins.explode() ]
    successes = dframe.groupby('task')['fun'].transform(lambda x: (x - x.min()) / (abs(x.min()) + 1))
    dframe['(f-opt)/(abs(opt)+1)'] = successes
    dframe['eps2'] = dframe.groupby('task')['fun2'].transform(lambda x: (x - x.min()) / (abs(x.min()) + 1))
    print(dframe)
    print('fun solved:')
    resss = dframe.groupby('solver').agg({'(f-opt)/(abs(opt)+1)': (lambda x: sum(x<1e-4))})
    print(resss)
    bad_f = dframe[(dframe['solver'] == TARGET_SOLVER) & (dframe['(f-opt)/(abs(opt)+1)'] >= 1e-4)]
    print('(f-opt)/(abs(opt)+1) >= 1e-4:')
    print(bad_f[['task', 'message', 'fun2', '(f-opt)/(abs(opt)+1)', 'pgnorm', 'eps2']])
    print('eps2 solved:')
    resss2 = dframe.groupby('solver').agg({'eps2': (lambda x: sum(x<1e-4))})
    print(resss2)
    bad = dframe[(dframe['solver'] == TARGET_SOLVER) & (dframe['eps2'] >=1e-4)]
    extended_bad = dframe[dframe['task'].isin(bad['task']) & (dframe['best'] | (dframe['solver'] == TARGET_SOLVER))]
    print('eps2 >= 1e-4:')
    print(bad[['task', 'message', 'fun2', 'pgnorm', 'eps2', 'local_mins_dist']])
    print(f'eps2 >= 1e-4 compared to best:')
    print(extended_bad[['task', 'solver', 'fun', 'fun2', 'pgnorm']])

    pg_rel_norm = dframe['pgnorm'] / (abs(dframe['fun2']) + 1)
    dframe['rel_pgnorm'] = pg_rel_norm
    local_res = dframe.groupby('solver').agg({ 'rel_pgnorm': (lambda x: sum(x<1e-6)) })
    print('rel pgnorm solved')
    print(local_res)
    pg_bad = dframe[(dframe['solver'] == TARGET_SOLVER) & (dframe['rel_pgnorm'] >= 1e-6)]
    extended_pg_bad = dframe[dframe['task'].isin(pg_bad['task']) & (dframe['best'] | (dframe['solver'] == TARGET_SOLVER))]
    print('rel_pgnorm >= 1e-6')
    print(pg_bad[['task', 'message', 'fun2', 'pgnorm', 'rel_pgnorm', 'eps2', 'local_mins_dist']])
    print('rel_pgnorm >= 1e-6 compared to best')
    print(extended_pg_bad[['task', 'solver', 'message', 'fun', 'rel_pgnorm', '(f-opt)/(abs(opt)+1)']])

    local_and_global_opt = (dframe['eps2'] < 1e-4) | (dframe['rel_pgnorm'] < 1e-6)
    dframe['both_opt'] = local_and_global_opt
    local_and_global_res = dframe.groupby('solver').agg({ 'both_opt': (lambda x: sum(1*x)) })
    print(local_and_global_res)
    bad_both = dframe[(dframe['solver'] == TARGET_SOLVER) & ~dframe['both_opt']]
    print('eps2 >= 1e-4 and rel_pgnorm >= 1e-6')
    print(bad_both[['task', 'message', 'nit', 'fun2', 'pgnorm', 'rel_pgnorm', 'eps2', 'local_mins_dist']])

    dframe['fev'] = dframe.apply(lambda x: np.argmax((np.array(x['fs']) - x['opt2'])/(1. + abs(x['opt2'])) < 1e-4) if any((np.array(x['fs']) - x['opt2'])/(1. + abs(x['opt2'])) < 1e-4) else int(1e9), axis=1) + 1
    dframe['pgev'] = dframe.apply(lambda x: np.argmax(x['pgs'] / (1. + abs(np.array(x['fs']))) < 1e-6) if any(x['pgs'] / (1. + abs(np.array(x['fs']))) < 1e-6) else int(1e9), axis=1) + 1
    dframe['nfev2'] = dframe[['fev', 'pgev']].min(axis=1)
    dframe['fun_final'] = dframe.apply(lambda x: x['fs'][(x['nfev2'] if x['nfev2'] < int(1e9) else len(x['fs'])) - 1] if len(x['fs']) > 0 else x['fun2'], axis=1)
    dframe['pg_final'] = dframe.apply(lambda x: x['pgs'][(x['nfev2'] if x['nfev2'] < int(1e9) else len(x['pgs'])) - 1] if len(x['pgs']) > 0 else x['pgnorm'], axis=1)
    
    print()
    print('Fixed iterations')
    print(dframe[['task', 'solver', 'fun2', 'pgnorm', 'nfev', 'fev', 'pgev', 'nfev2']])
    
    nfev_compare = dframe.groupby('task').filter(lambda x: x[x['solver'] == TARGET_SOLVER]['nfev2'].item() > x[x['solver'] == OTHER_SOLVER]['nfev2'].item())
    print()
    print(nfev_compare)

    available_problems = pycutest.find_problems(constraints='unconstrained')
    available_solvers = list(map(lambda f: f[:-3],
                                 filter(lambda f: f != '__init__.py' and f.endswith('.py'),
                                        os.listdir('optimization_solvers'))))

    available_problems = sorted(available_problems)
    available_solvers = sorted(available_solvers)

    if args.plot_again:
        metrics = ['nit', 'nfev', 'time']
        save_results(results, metrics, {}, {}, one_fig_pdf=(not args.single_fig_per_f_over_time), barplots_cnt=args.barplots_cnt) 
