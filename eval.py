import argparse
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import utils


plt.rcParams["figure.figsize"] = [8, 4]
plt.rcParams["figure.dpi"] = 100


def task_exp(args, extra_args):
    N_EXPS = args.nExps
    N_TASKS = args.nTasks
    N_ARMS = args.nArms
    HORIZON = args.horizon
    OPT_SIZE = args.optSize
    N_EXPERT = extra_args["nExperts"]
    if not args.loadCache:
        (X, regret_dict, title, xlabel, ylabel) = utils.task_exp(
            N_EXPS, N_TASKS, N_ARMS, HORIZON, OPT_SIZE, N_EXPERT, **extra_args
        )
    else:
        X = np.arange(N_TASKS)
        gap = extra_args["gap_constrain"]
        regret_dict = pickle.load(open(os.path.join(args.cacheDir, "tasks_cache.p"), "rb"))
        title = (
            f"Regret: {N_ARMS} arms, horizon {HORIZON}, {N_EXPERT} experts, gap = {gap:.3f} and subset size {OPT_SIZE}"
        )
        xlabel, ylabel = "Number of tasks", "Average Regret per task"
        indices = np.arange(0, X.shape[0], extra_args["task_cache_step"]).astype(int)
        utils.plot(X[indices], regret_dict, title, xlabel, ylabel, extra_args["plot_var"])
        plt.savefig(os.path.join(args.cacheDir, "task_exp.png"))
    pickle.dump(regret_dict, open(os.path.join(args.cacheDir, "tasks.p"), "wb"))


def horizon_exp(args, extra_args):
    N_EXPS = args.nExps
    N_TASKS = args.nTasks
    N_ARMS = args.nArms
    OPT_SIZE = args.optSize
    N_EXPERT = extra_args["nExperts"]
    if not args.loadCache:
        horizon_list = np.arange(extra_args["exp_args"][0], extra_args["exp_args"][1], extra_args["exp_args"][2])
        (X_h, regret_dict_h, title, xlabel, ylabel) = utils.horizon_exp(
            N_EXPS, N_TASKS, N_ARMS, OPT_SIZE, N_EXPERT, horizon_list=horizon_list, **extra_args
        )
    else:
        X_h = np.arange(50, 310, 50)
        regret_dict_h = pickle.load(open(os.path.join(args.cacheDir, "horizon_cache.p"), "rb"))
        title = f"Regret: {N_ARMS} arms, {N_TASKS} tasks, {N_EXPERT} experts, gap cond. satisfied and subset size {OPT_SIZE}"
        xlabel, ylabel = "Horizon", "Average Regret per Step"
        utils.plot(X_h, regret_dict_h, title, xlabel, ylabel, extra_args["plot_var"])
    pickle.dump(regret_dict_h, open(os.path.join(args.cacheDir, "horizon.p"), "wb"))


def subset_exp(args, extra_args):
    N_EXPS = args.nExps
    N_TASKS = args.nTasks
    N_ARMS = args.nArms
    HORIZON = args.horizon
    N_EXPERT = extra_args["nExperts"]
    if not args.loadCache:
        (X_e, regret_dict_e, title, xlabel, ylabel) = utils.subset_exp(
            N_EXPS, N_TASKS, N_ARMS, HORIZON, N_EXPERT, opt_size_list=np.arange(1, N_ARMS + 1, 1), **extra_args
        )
    else:
        gap = extra_args["gap_constrain"]
        title = f"Regret: {N_ARMS} arms, Horizon {HORIZON}, {N_TASKS} tasks, gap = {gap:.3f} and all experts"
        xlabel, ylabel = "subset size", "Regret"
        X_e = np.arange(1, N_ARMS + 1, 1)
        regret_dict_e = pickle.load(open(os.path.join(args.cacheDir, "subset_cache.p"), "rb"))
        utils.plot(X_e, regret_dict_e, title, xlabel, ylabel, extra_args["plot_var"])
    pickle.dump(regret_dict_e, open(os.path.join(args.cacheDir, "subset.p"), "wb"))


def arms_exp(args, extra_args):
    N_EXPS = args.nExps
    N_TASKS = args.nTasks
    HORIZON = args.horizon
    OPT_SIZE = args.optSize
    N_EXPERT = extra_args["nExperts"]
    if not args.loadCache:
        n_arms_list = np.arange(extra_args["exp_args"][0], extra_args["exp_args"][1], extra_args["exp_args"][2])
        (X_b, regret_dict_b, title, xlabel, ylabel) = utils.arms_exp(
            N_EXPS, N_TASKS, HORIZON, OPT_SIZE, N_EXPERT, n_arms_list, **extra_args
        )
    else:
        title = (
            f"Regret: Horizon {HORIZON}, {N_TASKS} tasks, all experts, gap cond. satisfied and subset size {OPT_SIZE}"
        )
        xlabel, ylabel = "Number of Arms", "Regret"
        X_b = np.arange(3, 8, 1)
        regret_dict_b = pickle.load(open(os.path.join(args.cacheDir, "arms_cache.p"), "rb"))
        utils.plot(X_b, regret_dict_b, title, xlabel, ylabel, extra_args["plot_var"])
    pickle.dump(regret_dict_b, open(os.path.join(args.cacheDir, "arms.p"), "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", help="choose experiment (task, horizon, arms, and subset)", type=str, default="task")
    parser.add_argument("--loadCache", dest="loadCache", action="store_true")
    parser.add_argument("--notLoadCache", dest="loadCache", action="store_false")
    parser.set_defaults(loadCache=False)
    parser.add_argument("--nTasks", help="number of tasks", type=int, default=1000)
    parser.add_argument("--nArms", help="number of arms", type=int, default=5)
    parser.add_argument("--nExps", help="number of repeated experiments", type=int, default=10)
    parser.add_argument("--optSize", help="size of the optimal subset", type=int, default=2)
    parser.add_argument("--horizon", help="horizon of each task", type=int, default=250)
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
        nExperts = 10
    else:
        if args.exp == "horizon" or args.exp == "arms":
            exp_args = json.loads(args.expArgs)
        nExperts = None
    GAP_THRESHOLD = min(1, np.sqrt(args.nArms * np.log(args.nTasks) / args.horizon) * 1.05)
    extra_args = {
        "exp_args": exp_args,
        "task_cache_step": 10,
        "gap_constrain": GAP_THRESHOLD,  # 1.05 is small gap, 1.2 for large
        "plot_var": True,
        "nExperts": nExperts,
    }

    if args.exp == "task":
        task_exp(args, extra_args)
    elif args.exp == "horizon":
        horizon_exp(args, extra_args)
    elif args.exp == "arms":
        arms_exp(args, extra_args)
    elif args.exp == "subset":
        subset_exp(args, extra_args)
