import inspect

import algos
import bandit
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from copy import deepcopy
import time


TASK_EXP = 0
HORIZON_EXP = 1
ARM_EXP = 2
SUBSET_EXP = 3


def rolls_out(agent, env, horizon, quiet):
    """
    Rolls-out 1 task
    """
    sum_rewards = 0
    obs = env.reset()
    if quiet is False:
        for i in trange(horizon):
            a = agent.get_action(obs)
            next_obs, r, _, _ = env.step(a)
            sum_rewards += r
            obs = next_obs
            if hasattr(agent, "update"):
                agent.update(a, r)
    else:
        for i in range(horizon):
            a = agent.get_action(obs)
            next_obs, r, _, _ = env.step(a)
            sum_rewards += r
            obs = next_obs
            if hasattr(agent, "update"):
                agent.update(a, r)
    if hasattr(agent, "eps_end_update"):
        agent.eps_end_update(obs)
    agent.reset()
    regret = np.max(env._p) * horizon - sum_rewards
    return regret


def meta_rolls_out(n_tasks, agent, env, horizon, timeout=None, **kwargs):
    """
    Rolls-out n_tasks
    """
    regrets = []
    EXT_set = [] # For adversarial setting
    start_time = time.time()
    for idx in range(n_tasks):
        env.reset_task(idx)
        r = rolls_out(agent, env, horizon, kwargs['quiet'])
        regrets.append(r)
        if kwargs['is_adversarial'] is True:
            if isinstance(agent, algos.ExpertMOSS): # Optimal MOSS
                EXT_set = env.opt_indices
            elif isinstance(agent, algos.PhaseElim):
                surviving_arms = agent.A_l
                EXT_set = list(set(EXT_set+surviving_arms))
            elif isinstance(agent, algos.MOSS):
                mu = deepcopy(env.counter)[::2]
                opt_idx = np.argmax(mu)
                EXT_set.append(opt_idx)
                EXT_set = list(set(EXT_set))
            else:
                EXT_set = agent.EXT_set

            env.generate_next_task(EXT_set)
        if timeout is not None and timeout*60 < time.time() - start_time:
            print('Timeout! Ending this task ...')
            break
    return regrets


def plot(X, regret_dict, title, xlabel, ylabel, **kwargs):
    moss_regrets = regret_dict["moss_regrets"]
    EE_regrets = regret_dict["EE_regrets"]
    opt_moss_regrets = regret_dict["opt_moss_regrets"]
    GML_regrets = regret_dict["GML_regrets"]

    moss_Y = np.mean(moss_regrets, axis=0)
    EE_Y = np.mean(EE_regrets, axis=0)
    opt_moss_Y = np.mean(opt_moss_regrets, axis=0)
    GML_Y = np.mean(GML_regrets, axis=0)

    # Unfinished runs have regret == -1
    moss_max_idx = max(np.where(moss_regrets!=-1)[1])+1
    EE_max_idx = max(np.where(EE_regrets!=-1)[1])+1
    opt_moss_max_idx = max(np.where(opt_moss_regrets!=-1)[1])+1
    GML_max_idx = max(np.where(GML_regrets!=-1)[1])+1

    if "PMML" not in kwargs['skip_list']:
        PMML_regrets = regret_dict["PMML_regrets"]
        PMML_Y = np.mean(PMML_regrets, axis=0)
        PMML_max_idx = max(np.where(PMML_regrets!=-1)[1])+1

    if kwargs['plot_var'] is True:
        moss_dY = 2 * np.sqrt(np.var(moss_regrets, axis=0))
        EE_dY = 2 * np.sqrt(np.var(EE_regrets, axis=0))
        opt_moss_dY = 2 * np.sqrt(np.var(opt_moss_regrets, axis=0))
        GML_dY = 2 * np.sqrt(np.var(GML_regrets, axis=0))

        plt.errorbar(X[:moss_max_idx], moss_Y[:moss_max_idx], moss_dY[:moss_max_idx], fmt="-", color="green", label="MOSS")
        plt.errorbar(X[:EE_max_idx], EE_Y[:EE_max_idx], EE_dY[:EE_max_idx], fmt="-", color="blue", label="EE")
        plt.errorbar(X[:opt_moss_max_idx], opt_moss_Y[:opt_moss_max_idx], opt_moss_dY[:opt_moss_max_idx], fmt="-", color="black", label="Optimal MOSS")
        plt.errorbar(X[:GML_max_idx], GML_Y[:GML_max_idx], GML_dY[:GML_max_idx], fmt="-", color="purple", label="GML")
        if "PMML" not in kwargs['skip_list']:
            PMML_dY = 2 * np.sqrt(np.var(PMML_regrets, axis=0))
            plt.errorbar(X[:PMML_max_idx], PMML_Y[:PMML_max_idx], PMML_dY[:PMML_max_idx], fmt="-", color="red", label="PMML")
    else:
        plt.plot(X[:moss_max_idx], moss_Y[:moss_max_idx], "-", color="green", label="MOSS")
        plt.plot(X[:EE_max_idx], EE_Y[:EE_max_idx], "-", color="blue", label="EE")
        plt.plot(X[:opt_moss_max_idx], opt_moss_Y[:opt_moss_max_idx], "-", color="black", label="Optimal MOSS")
        plt.plot(X[:GML_max_idx], GML_Y[:GML_max_idx], "-", color="purple", label="GML")
        if "PMML" not in kwargs['skip_list']:
            plt.plot(X[:PMML_max_idx], PMML_Y[:PMML_max_idx], "-", color="red", label="PMML")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)
    caller_name = inspect.stack()[1][3]
    plt.savefig(f"./results/{caller_name}.png")


def _init_agents(N_EXPS, N_TASKS, N_ARMS, HORIZON, OPT_SIZE, env, **kwargs):
    moss_agent = algos.MOSS(n_arms=N_ARMS, horizon=HORIZON)
    EE_agent = algos.EE(n_arms=N_ARMS, horizon=HORIZON, n_tasks=N_TASKS, subset_size=OPT_SIZE)
    opt_moss_agent = algos.ExpertMOSS(n_arms=N_ARMS, horizon=HORIZON, expert_subset=env.opt_indices)
    GML_agent = algos.GML(n_arms=N_ARMS, horizon=HORIZON, n_tasks=N_TASKS)
    output = {
        "moss_agent": moss_agent,
        "EE_agent": EE_agent,
        "opt_moss_agent": opt_moss_agent,
        "GML_agent": GML_agent,
    }
    if "PMML" not in kwargs['skip_list']:
        PMML_agent = algos.PMML(n_arms=N_ARMS, horizon=HORIZON, n_tasks=N_TASKS, subset_size=OPT_SIZE)
        output["PMML_agent"] = PMML_agent
    return output


def _init_cache(N_EXPS, x_axis):
    moss_regrets = np.zeros((N_EXPS, x_axis))-1
    EE_regrets = np.zeros((N_EXPS, x_axis))-1
    PMML_regrets = np.zeros((N_EXPS, x_axis))-1
    opt_moss_regrets = np.zeros((N_EXPS, x_axis))-1
    GML_regrets = np.zeros((N_EXPS, x_axis))-1
    return {
        "moss_regrets": moss_regrets,
        "EE_regrets": EE_regrets,
        "PMML_regrets": PMML_regrets,
        "opt_moss_regrets": opt_moss_regrets,
        "GML_regrets": GML_regrets,
    }


def _collect_data(agent_dict, cache_dict, i, j, n_tasks, HORIZON, env, exp_type, timer_cache, **kwargs):
    def _rolls_out_and_time(name, timer_cache):
        tic = time.time()
        r = None
        if timer_cache is None or timer_cache['timeout']*60 > timer_cache[name]:
            tmp_dict = {'quiet':kwargs['quiet'], 'is_adversarial':kwargs['is_adversarial']}
            # using tmp_dict instead of kwargs to specify 'timeout' = None
            r = meta_rolls_out(n_tasks, agent_dict[name+"_agent"], env, HORIZON, timeout=None, **tmp_dict)
        else:
            print('Timeout! Ending this task ...')
        toc = time.time()
        timer_cache[name] += toc - tic
        return timer_cache, r

    if exp_type == TASK_EXP:
        moss_r = meta_rolls_out(n_tasks, agent_dict["moss_agent"], env, HORIZON, **kwargs)
        EE_r = meta_rolls_out(n_tasks, agent_dict["EE_agent"], env, HORIZON, **kwargs)
        if "PMML" not in kwargs['skip_list']:
            PMML_r = meta_rolls_out(n_tasks, agent_dict["PMML_agent"], env, HORIZON, **kwargs)
        opt_moss_r = meta_rolls_out(n_tasks, agent_dict["opt_moss_agent"], env, HORIZON, **kwargs)
        GML_r = meta_rolls_out(n_tasks, agent_dict["GML_agent"], env, HORIZON, **kwargs)
    else:
        timer_cache, moss_r = _rolls_out_and_time("moss", timer_cache)
        timer_cache, EE_r = _rolls_out_and_time("EE", timer_cache)
        if "PMML" not in kwargs['skip_list']:
            timer_cache, PMML_r = _rolls_out_and_time("PMML", timer_cache)
        timer_cache, opt_moss_r = _rolls_out_and_time("opt_moss", timer_cache)
        timer_cache, GML_r = _rolls_out_and_time("GML", timer_cache)
    if exp_type == TASK_EXP:
        if moss_r is not None: cache_dict["moss_regrets"][i,:len(moss_r)] = moss_r
        if EE_r is not None: cache_dict["EE_regrets"][i,:len(EE_r)] = EE_r
        if "PMML" not in kwargs['skip_list'] and PMML_r is not None:
            cache_dict["PMML_regrets"][i,:len(PMML_r)] = PMML_r
        if opt_moss_r is not None: cache_dict["opt_moss_regrets"][i,:len(opt_moss_r)] = opt_moss_r
        if GML_r is not None: cache_dict["GML_regrets"][i,:len(GML_r)] = GML_r
    elif exp_type == HORIZON_EXP:
        if moss_r is not None: cache_dict["moss_regrets"][i, j] = np.mean(moss_r) / HORIZON
        if EE_r is not None: cache_dict["EE_regrets"][i, j] = np.mean(EE_r) / HORIZON
        if "PMML" not in kwargs['skip_list'] and PMML_r is not None:
            cache_dict["PMML_regrets"][i, j] = np.mean(PMML_r) / HORIZON
        if opt_moss_r is not None: cache_dict["opt_moss_regrets"][i, j] = np.mean(opt_moss_r) / HORIZON
        if GML_r is not None: cache_dict["GML_regrets"][i, j] = np.mean(GML_r) / HORIZON
    else:
        if moss_r is not None: cache_dict["moss_regrets"][i, j] = np.mean(moss_r)
        if EE_r is not None: cache_dict["EE_regrets"][i, j] = np.mean(EE_r)
        if "PMML" not in kwargs['skip_list'] and PMML_r is not None:
            cache_dict["PMML_regrets"][i, j] = np.mean(PMML_r)
        if opt_moss_r is not None: cache_dict["opt_moss_regrets"][i, j] = np.mean(opt_moss_r)
        if GML_r is not None: cache_dict["GML_regrets"][i, j] = np.mean(GML_r)
    return cache_dict, timer_cache


def task_exp(N_EXPS, N_TASKS, N_ARMS, HORIZON, OPT_SIZE, **kwargs):
    if kwargs['is_adversarial'] is False:
        setting = "Stochastic setting"
        env = bandit.MetaBernoulli(n_arms=N_ARMS, opt_size=OPT_SIZE, n_tasks=N_TASKS, **kwargs)
    else:
        setting = "Adversarial setting"
        env = bandit.AdvMetaBernoulli(n_arms=N_ARMS, opt_size=OPT_SIZE, n_tasks=N_TASKS, horizon=HORIZON, **kwargs)
    cache_dict = _init_cache(N_EXPS, N_TASKS)
    for i in trange(N_EXPS):
        agent_dict = _init_agents(N_EXPS, N_TASKS, N_ARMS, HORIZON, OPT_SIZE, env, **kwargs)
        cache_dict, timer_cache = _collect_data(agent_dict, cache_dict, i, None, N_TASKS, HORIZON, env, TASK_EXP, {'timeout':kwargs['timeout']}, **kwargs)
    X = np.arange(N_TASKS)
    gap = kwargs["gap_constrain"]
    title = f"Regret: {setting}, {N_ARMS} arms, horizon {HORIZON}, {int(env.n_experts)} experts, gap = {gap:.3f} and subset size {OPT_SIZE}"
    xlabel, ylabel = "Number of tasks", "Average Regret per task"
    step = kwargs["task_cache_step"]
    indices = np.arange(0, X.shape[0], step).astype(int)
    cache_dict["moss_regrets"] = cache_dict["moss_regrets"][:, indices]
    cache_dict["EE_regrets"] = cache_dict["EE_regrets"][:, indices]
    if "PMML" not in kwargs['skip_list']:
        cache_dict["PMML_regrets"] = cache_dict["PMML_regrets"][:, indices]
    cache_dict["opt_moss_regrets"] = cache_dict["opt_moss_regrets"][:, indices]
    cache_dict["GML_regrets"] = cache_dict["GML_regrets"][:, indices]
    plot(X[indices], cache_dict, title, xlabel, ylabel, **kwargs)
    return (X, cache_dict, title, xlabel, ylabel)


def horizon_exp(
    N_EXPS,
    N_TASKS,
    N_ARMS,
    OPT_SIZE,
    horizon_list=np.arange(1, 202, 50) * 10,
    **kwargs,
):
    cache_dict = _init_cache(N_EXPS, horizon_list.shape[0])
    for i in trange(N_EXPS):
        timer_cache = {'timeout': kwargs['timeout'], "moss":0, "EE":0, "PMML":0, "opt_moss":0, "GML":0,}
        for j, h in enumerate(horizon_list):
            kwargs["gap_constrain"] = min(1, np.sqrt(N_ARMS * np.log(N_TASKS) / h))
            tmp = kwargs["gap_constrain"]
            print(f"gap = {tmp}")
            if kwargs['is_adversarial'] is False:
                env = bandit.MetaBernoulli(n_arms=N_ARMS, opt_size=OPT_SIZE, n_tasks=N_TASKS, **kwargs)
            else:
                env = bandit.AdvMetaBernoulli(n_arms=N_ARMS, opt_size=OPT_SIZE, n_tasks=N_TASKS, horizon=h, **kwargs)
            agent_dict = _init_agents(N_EXPS, N_TASKS, N_ARMS, h, OPT_SIZE, env, **kwargs)
            cache_dict, timer_cache = _collect_data(agent_dict, cache_dict, i, j, N_TASKS, h, env, HORIZON_EXP, timer_cache, **kwargs)
    X = horizon_list
    if kwargs['is_adversarial'] is False:
        setting = "Stochastic setting"
    else:
        setting = "Adversarial setting"
    title = f"Regret: {setting}, {N_ARMS} arms, {N_TASKS} tasks, all experts, gap cond. satisfied and subset size {OPT_SIZE}"
    xlabel, ylabel = "Horizon", "Average Regret per Step"
    plot(X, cache_dict, title, xlabel, ylabel, **kwargs)
    return (X, cache_dict, title, xlabel, ylabel)


def arms_exp(N_EXPS, N_TASKS, HORIZON, OPT_SIZE, n_arms_list=np.arange(8, 69, 15), **kwargs):
    cache_dict = _init_cache(N_EXPS, n_arms_list.shape[0])
    for i in trange(N_EXPS):
        timer_cache = {'timeout': kwargs['timeout'], "moss":0, "EE":0, "PMML":0, "opt_moss":0, "GML":0,}
        for j, b in enumerate(n_arms_list):
            kwargs["gap_constrain"] = min(1, np.sqrt(b * np.log(N_TASKS) / HORIZON))
            if kwargs['is_adversarial'] is False:
                env = bandit.MetaBernoulli(n_arms=b, opt_size=OPT_SIZE, n_tasks=N_TASKS, **kwargs)
            else:
                env = bandit.AdvMetaBernoulli(n_arms=b, opt_size=OPT_SIZE, n_tasks=N_TASKS, horizon=HORIZON, **kwargs)
            agent_dict = _init_agents(N_EXPS, N_TASKS, b, HORIZON, OPT_SIZE, env, **kwargs)
            cache_dict, timer_cache = _collect_data(agent_dict, cache_dict, i, j, N_TASKS, HORIZON, env, ARM_EXP, timer_cache, **kwargs)
    X = n_arms_list
    if kwargs['is_adversarial'] is False:
        setting = "Stochastic setting"
    else:
        setting = "Adversarial setting"
    title = f"Regret: {setting}, Horizon {HORIZON}, {N_TASKS} tasks, all experts, gap cond. satisfied and subset size {OPT_SIZE}"
    xlabel, ylabel = "Number of Arms", "Regret"
    plot(X, cache_dict, title, xlabel, ylabel, **kwargs)
    return (X, cache_dict, title, xlabel, ylabel)


def subset_exp(N_EXPS, N_TASKS, N_ARMS, HORIZON, opt_size_list=None, **kwargs):
    if opt_size_list is None:
        opt_size_list = np.arange(1, N_ARMS + 1, 4)
    cache_dict = _init_cache(N_EXPS, opt_size_list.shape[0])
    for i in trange(N_EXPS):
        timer_cache = {'timeout': kwargs['timeout'], "moss":0, "EE":0, "PMML":0, "opt_moss":0, "GML":0,}
        for j, s in enumerate(opt_size_list):
            if kwargs['is_adversarial'] is False:
                env = bandit.MetaBernoulli(n_arms=N_ARMS, opt_size=s, n_tasks=N_TASKS, **kwargs)
            else:
                env = bandit.AdvMetaBernoulli(n_arms=N_ARMS, opt_size=s, n_tasks=N_TASKS, horizon=HORIZON, **kwargs)
            agent_dict = _init_agents(N_EXPS, N_TASKS, N_ARMS, HORIZON, s, env, **kwargs)
            cache_dict, timer_cache = _collect_data(agent_dict, cache_dict, i, j, N_TASKS, HORIZON, env, SUBSET_EXP, timer_cache, **kwargs)
    X = opt_size_list
    gap = kwargs["gap_constrain"]
    if kwargs['is_adversarial'] is False:
        setting = "Stochastic setting"
    else:
        setting = "Adversarial setting"
    title = f"Regret: {setting}, {N_ARMS} arms, Horizon {HORIZON}, {N_TASKS} tasks, gap = {gap:.3f} and all experts"
    xlabel, ylabel = "subset size", "Regret"
    plot(X, cache_dict, title, xlabel, ylabel, **kwargs)
    return (X, cache_dict, title, xlabel, ylabel)
