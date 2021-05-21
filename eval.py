import argparse
import json
import os
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np
import utils


font = {
#         'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

plt.rc('font', **font)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.figsize'] = [9, 5.5] # NIPS format: [9, 5.5]
plt.rcParams['figure.dpi'] = 300


def task_exp(args, extra_args):
    N_EXPS = args.nExps
    N_TASKS = args.nTasks
    N_ARMS = args.nArms
    HORIZON = args.horizon
    OPT_SIZE = args.optSize
    if extra_args['is_adversarial']:
        setting = "Adversarial"
    else:
        setting = "Stochastic"
    if not args.loadCache:
        task_list = np.arange(extra_args["exp_args"][0], extra_args["exp_args"][1], extra_args["exp_args"][2])
        (X, regret_dict, title, xlabel, ylabel) = utils.task_exp(N_EXPS, HORIZON, N_ARMS, OPT_SIZE, task_list, **extra_args)
    else:
        X = np.arange(N_TASKS)
        gap = extra_args["gap_constrain"]
        regret_dict = pickle.load(open(os.path.join(args.cacheDir, "tasks_cache.p"), "rb"))
        title = f'{setting}:{N_ARMS} arms, horizon = {HORIZON}, and subset size = {OPT_SIZE}'
        xlabel, ylabel = "Number of tasks", "Average Regret per task"
        indices = np.arange(0, X.shape[0], extra_args["task_cache_step"]).astype(int)
        utils.plot(X[indices], regret_dict, title, xlabel, ylabel, extra_args["plot_var"])
        plt.savefig(os.path.join(args.cacheDir, "task_exp.png"))
    pickle.dump(regret_dict, open(os.path.join(args.cacheDir, setting+"_tasks.p"), "wb"))


def horizon_exp(args, extra_args):
    N_EXPS = args.nExps
    N_TASKS = args.nTasks
    N_ARMS = args.nArms
    OPT_SIZE = args.optSize
    if extra_args['is_adversarial']:
        setting = "Adversarial"
    else:
        setting = "Stochastic"
    if not args.loadCache:
        horizon_list = np.arange(extra_args["exp_args"][0], extra_args["exp_args"][1], extra_args["exp_args"][2])
        (X_h, regret_dict_h, title, xlabel, ylabel) = utils.horizon_exp(
            N_EXPS, N_TASKS, N_ARMS, OPT_SIZE, horizon_list=horizon_list, **extra_args
        )
    else:
        X_h = np.arange(50, 310, 50)
        regret_dict_h = pickle.load(open(os.path.join(args.cacheDir, "horizon_cache.p"), "rb"))
        title = f'{setting}: {N_ARMS} arms, {N_TASKS} tasks, and subset size = {OPT_SIZE}'
        xlabel, ylabel = "Horizon", "Average Regret per Step"
        utils.plot(X_h, regret_dict_h, title, xlabel, ylabel, extra_args["plot_var"])
    pickle.dump(regret_dict_h, open(os.path.join(args.cacheDir, setting+"_horizon.p"), "wb"))


def subset_exp(args, extra_args):
    N_EXPS = args.nExps
    N_TASKS = args.nTasks
    N_ARMS = args.nArms
    HORIZON = args.horizon
    if extra_args['is_adversarial']:
        setting = "Adversarial"
    else:
        setting = "Stochastic"
    if extra_args["exp_args"] is None:
        X_e = np.arange(2, N_ARMS + 1, 1)
    else:
        X_e = np.arange(extra_args["exp_args"][0], extra_args["exp_args"][1], extra_args["exp_args"][2])
    if not args.loadCache:
        (X_e, regret_dict_e, title, xlabel, ylabel) = utils.subset_exp(
            N_EXPS, N_TASKS, N_ARMS, HORIZON, opt_size_list=X_e, **extra_args
        )
    else:
        gap = extra_args["gap_constrain"]
        title = f'{setting}: {N_ARMS} arms, horizon = {HORIZON}, {N_TASKS} tasks'
        xlabel, ylabel = "subset size", "Regret"
        regret_dict_e = pickle.load(open(os.path.join(args.cacheDir, "subset_cache.p"), "rb"))
        utils.plot(X_e, regret_dict_e, title, xlabel, ylabel, extra_args["plot_var"])
    pickle.dump(regret_dict_e, open(os.path.join(args.cacheDir, setting+"_subset.p"), "wb"))


def arms_exp(args, extra_args):
    N_EXPS = args.nExps
    N_TASKS = args.nTasks
    HORIZON = args.horizon
    OPT_SIZE = args.optSize
    if extra_args['is_adversarial']:
        setting = "Adversarial"
    else:
        setting = "Stochastic"
    if not args.loadCache:
        n_arms_list = np.arange(extra_args["exp_args"][0], extra_args["exp_args"][1], extra_args["exp_args"][2])
        (X_b, regret_dict_b, title, xlabel, ylabel) = utils.arms_exp(
            N_EXPS, N_TASKS, HORIZON, OPT_SIZE, n_arms_list, **extra_args
        )
    else:
        title = f'{setting}: Horizon = {HORIZON}, {N_TASKS} tasks, and subset size = {OPT_SIZE}'
        xlabel, ylabel = "Number of Arms", "Regret"
        X_b = np.arange(3, 8, 1)
        regret_dict_b = pickle.load(open(os.path.join(args.cacheDir, "arms_cache.p"), "rb"))
        utils.plot(X_b, regret_dict_b, title, xlabel, ylabel, extra_args["plot_var"])
    pickle.dump(regret_dict_b, open(os.path.join(args.cacheDir, setting+"_arms.p"), "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", help="choose experiment (task, horizon, arms, and subset)", type=str, default="task")
    parser.add_argument("--loadCache", dest="loadCache", action="store_true")
    parser.add_argument("--notLoadCache", dest="loadCache", action="store_false")
    parser.set_defaults(loadCache=False)
    parser.add_argument("--adversarial", dest="isAdversarial", action="store_true")
    parser.add_argument("--stochastic", dest="isAdversarial", action="store_false")
    parser.set_defaults(isAdversarial=True)
    parser.add_argument("--quiet", dest="quiet", action="store_true")
    parser.add_argument("--notQuiet", dest="quiet", action="store_false")
    parser.set_defaults(quiet=True)
    parser.add_argument("--nTasks", help="number of tasks", type=int, default=1000)
    parser.add_argument("--nArms", help="number of arms", type=int, default=5)
    parser.add_argument("--nExps", help="number of repeated experiments", type=int, default=10)
    parser.add_argument("--optSize", help="size of the optimal subset (must >1)", type=int, default=2)
    parser.add_argument("--horizon", help="horizon of each task", type=int, default=250)
    parser.add_argument("--timeOut", help="maximum minutes (for all settings) per experiment (total time divided by (repeat_exps * num_tested_method))", type=float, default=2)
    parser.add_argument(
        "--expArgs", help="arguments for horizon or arms. Example: (a,b,c) => range(a,b,c)", type=str, default=None
    )
    parser.add_argument("--cacheDir", help="directory of cache results", type=str, default="./results")
    parser.add_argument("--seed", help="seed number", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
    exp_args = None
    if args.loadCache is True:
        if (
            args.nTasks != 1000
            or args.nArms != 5
            or args.nExps != 10
            or args.optSize != 2
            or args.horizon != 250
            or args.expArgs is not None
        ):
            assert (
                False
            ), "When using loadCache, please use the default setting for nTasks, nArms, nExps, optSize, and horizon."
    GAP_THRESHOLD = np.sqrt(args.nArms * np.log(args.nTasks) / args.horizon)
    extra_args = {
        "exp_args": json.loads(args.expArgs),
        "task_cache_step": 20,
        "gap_constrain": min(1,GAP_THRESHOLD*1.4),  # 1.0005 is small gap, 1.2 for large
        "plot_var": True,
        "is_adversarial": args.isAdversarial,
        "timeout": args.timeOut, # maximum duration for each roll-outs. Unit = minute. -1 = unlimited
        "quiet": args.quiet,
        "skip_list": [],
#         "skip_list": ["PMML"],
        "linewidth": 4,
        "plot_legend": True,
    }
    tik = time.time()
    if args.exp == "task":
        task_exp(args, extra_args)
    elif args.exp == "horizon":
        horizon_exp(args, extra_args)
    elif args.exp == "arms":
        arms_exp(args, extra_args)
    elif args.exp == "subset":
        subset_exp(args, extra_args)
    tok = time.time()
    print(f"Total time spent: {(tok-tik)/3600} hours.")
