import algos
import bandit
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange


TASK_EXP = 0
HORIZON_EXP = 1
ARM_EXP = 2
SUBSET_EXP = 3


def rolls_out(agent, env, horizon, quiet):
    """
    Rolls-out 1 task
    """
    rewards = []
    obs = env.reset()
    if quiet is False:
        for i in trange(horizon):
            a = agent.get_action(obs)
            next_obs, r, _, _ = env.step(a)
            rewards.append(r)
            obs = next_obs
            if hasattr(agent, "update"):
                agent.update(a, r)
    else:
        for i in range(horizon):
            a = agent.get_action(obs)
            next_obs, r, _, _ = env.step(a)
            rewards.append(r)
            obs = next_obs
            if hasattr(agent, "update"):
                agent.update(a, r)
    if hasattr(agent, "eps_end_update"):
        agent.eps_end_update(obs)
    regret = np.max(env._p) * horizon - np.sum(rewards)
    return regret


def meta_rolls_out(n_tasks, agent, env, horizon, quiet):
    """
    Rolls-out n_tasks
    """
    regrets = []
    tmp_regrets = []
    for idx in range(n_tasks):
        env.reset_task(idx)
        r = rolls_out(agent, env, horizon, quiet)
        tmp_regrets.append(r)
        regrets.append(np.average(tmp_regrets))  # average regret until this task
    return regrets


def plot(X, regret_dict, title, xlabel, ylabel, plot_var=False):
    moss_regrets = regret_dict["moss_regrets"]
    EE_regrets = regret_dict["EE_regrets"]
    PMML_regrets = regret_dict["PMML_regrets"]
    opt_moss_regrets = regret_dict["opt_moss_regrets"]

    moss_Y = np.mean(moss_regrets, axis=0)
    EE_Y = np.mean(EE_regrets, axis=0)
    PMML_Y = np.mean(PMML_regrets, axis=0)
    opt_moss_Y = np.mean(opt_moss_regrets, axis=0)
    if plot_var is True:
        moss_dY = 2 * np.sqrt(np.var(moss_regrets, axis=0))
        EE_dY = 2 * np.sqrt(np.var(EE_regrets, axis=0))
        PMML_dY = 2 * np.sqrt(np.var(PMML_regrets, axis=0))
        opt_moss_dY = 2 * np.sqrt(np.var(opt_moss_regrets, axis=0))

        plt.errorbar(X, moss_Y, moss_dY, fmt="-", color="green", label="MOSS")
        plt.errorbar(X, EE_Y, EE_dY, fmt="-", color="blue", label="EE")
        plt.errorbar(X, PMML_Y, PMML_dY, fmt="-", color="red", label="PMML")
        plt.errorbar(X, opt_moss_Y, opt_moss_dY, fmt="-", color="black", label="Optimal MOSS")
    else:
        plt.plot(X, moss_Y, "-", color="green", label="MOSS")
        plt.plot(X, EE_Y, "-", color="blue", label="EE")
        plt.plot(X, PMML_Y, "-", color="red", label="PMML")
        plt.plot(X, opt_moss_Y, "-", color="black", label="Optimal MOSS")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)


def _init_agents(N_EXPS, N_TASKS, N_BANDITS, HORIZON, OPT_SIZE, N_EXPERT, DS_NAME, env, quiet=True, **kwargs):
    moss_agent = algos.MOSS(n_bandits=N_BANDITS, horizon=HORIZON)
    EE_agent = algos.EE(n_bandits=N_BANDITS, horizon=HORIZON, n_tasks=N_TASKS, expert_subsets=env.expert_subsets)
    PMML_agent = algos.PMML(n_bandits=N_BANDITS, horizon=HORIZON, n_tasks=N_TASKS, expert_subsets=env.expert_subsets)
    opt_moss_agent = algos.ExpertMOSS(n_bandits=N_BANDITS, horizon=HORIZON, expert_subset=env.opt_indices)
    return {
        "moss_agent": moss_agent,
        "EE_agent": EE_agent,
        "opt_moss_agent": opt_moss_agent,
        "PMML_agent": PMML_agent,
    }


def _init_cache(N_EXPS, x_axis):
    moss_regrets = np.zeros((N_EXPS, x_axis))
    EE_regrets = np.zeros((N_EXPS, x_axis))
    PMML_regrets = np.zeros((N_EXPS, x_axis))
    opt_moss_regrets = np.zeros((N_EXPS, x_axis))
    return {
        "moss_regrets": moss_regrets,
        "EE_regrets": EE_regrets,
        "PMML_regrets": PMML_regrets,
        "opt_moss_regrets": opt_moss_regrets,
    }


def _collect_data(agent_dict, cache_dict, i, j, n_tasks, HORIZON, quiet, env, exp_type):
    moss_r = meta_rolls_out(n_tasks, agent_dict["moss_agent"], env, HORIZON, quiet)
    EE_r = meta_rolls_out(n_tasks, agent_dict["EE_agent"], env, HORIZON, quiet)
    PMML_r = meta_rolls_out(n_tasks, agent_dict["PMML_agent"], env, HORIZON, quiet)
    opt_moss_r = meta_rolls_out(n_tasks, agent_dict["opt_moss_agent"], env, HORIZON, quiet)
    if exp_type == TASK_EXP:
        cache_dict["moss_regrets"][i] = moss_r
        cache_dict["EE_regrets"][i] = EE_r
        cache_dict["PMML_regrets"][i] = PMML_r
        cache_dict["opt_moss_regrets"][i] = opt_moss_r
    elif exp_type == HORIZON_EXP:
        cache_dict["moss_regrets"][i, j] = moss_r[-1] / HORIZON
        cache_dict["EE_regrets"][i, j] = EE_r[-1] / HORIZON
        cache_dict["PMML_regrets"][i, j] = PMML_r[-1] / HORIZON
        cache_dict["opt_moss_regrets"][i, j] = opt_moss_r[-1] / HORIZON
    else:
        cache_dict["moss_regrets"][i, j] = moss_r[-1]
        cache_dict["EE_regrets"][i, j] = EE_r[-1]
        cache_dict["PMML_regrets"][i, j] = PMML_r[-1]
        cache_dict["opt_moss_regrets"][i, j] = opt_moss_r[-1]
    return cache_dict


def task_exp(N_EXPS, N_TASKS, N_BANDITS, HORIZON, OPT_SIZE, N_EXPERT, DS_NAME, quiet=True, **kwargs):
    env = bandit.MetaBernoulli(
        n_bandits=N_BANDITS, opt_size=OPT_SIZE, n_tasks=N_TASKS, n_experts=N_EXPERT, ds_name=DS_NAME, **kwargs
    )
    cache_dict = _init_cache(N_EXPS, N_TASKS)
    for i in trange(N_EXPS):
        agent_dict = _init_agents(
            N_EXPS, N_TASKS, N_BANDITS, HORIZON, OPT_SIZE, N_EXPERT, DS_NAME, env, quiet, **kwargs
        )
        cache_dict = _collect_data(agent_dict, cache_dict, i, None, N_TASKS, HORIZON, quiet, env, TASK_EXP)
    X = np.arange(N_TASKS)
    if N_EXPERT is None:
        N_EXPERT = env.n_experts
    gap = kwargs["gap_constrain"]
    title = (
        f"Regret: {N_BANDITS} arms, horizon {HORIZON}, {N_EXPERT} experts, gap = {gap:.3f} and subset size {OPT_SIZE}"
    )
    xlabel, ylabel = "Number of tasks", "Average Regret per task"
    step = kwargs["task_cache_step"]
    indices = np.arange(0, X.shape[0], step).astype(int)
    regret_dict = {
        "moss_regrets": cache_dict["moss_regrets"][:, indices],
        "EE_regrets": cache_dict["EE_regrets"][:, indices],
        "PMML_regrets": cache_dict["PMML_regrets"][:, indices],
        "opt_moss_regrets": cache_dict["opt_moss_regrets"][:, indices],
    }
    plot(X[indices], regret_dict, title, xlabel, ylabel, kwargs["plot_var"])
    return (X, regret_dict, title, xlabel, ylabel)


def horizon_exp(
    N_EXPS,
    N_TASKS,
    N_BANDITS,
    OPT_SIZE,
    N_EXPERT,
    DS_NAME,
    horizon_list=np.arange(1, 202, 50) * 10,
    quiet=True,
    **kwargs,
):
    cache_dict = _init_cache(N_EXPS, horizon_list.shape[0])
    for i in trange(N_EXPS):
        for j, h in enumerate(horizon_list):
            kwargs["gap_constrain"] = min(1, np.sqrt(N_BANDITS * np.log(N_TASKS) / h))
            tmp = kwargs["gap_constrain"]
            print(f"gap = {tmp}")
            env = bandit.MetaBernoulli(
                n_bandits=N_BANDITS, opt_size=OPT_SIZE, n_tasks=N_TASKS, n_experts=N_EXPERT, ds_name=DS_NAME, **kwargs
            )
            agent_dict = _init_agents(N_EXPS, N_TASKS, N_BANDITS, h, OPT_SIZE, N_EXPERT, DS_NAME, env, quiet, **kwargs)
            cache_dict = _collect_data(agent_dict, cache_dict, i, j, N_TASKS, h, quiet, env, HORIZON_EXP)
    X = horizon_list
    if N_EXPERT is None:
        N_EXPERT = env.n_experts
    title = f"Regret: {N_BANDITS} arms, {N_TASKS} tasks, {N_EXPERT} experts, gap cond. satisfied and subset size {OPT_SIZE}"
    xlabel, ylabel = "Horizon", "Average Regret per Step"
    regret_dict = {
        "moss_regrets": cache_dict["moss_regrets"],
        "EE_regrets": cache_dict["EE_regrets"],
        "PMML_regrets": cache_dict["PMML_regrets"],
        "opt_moss_regrets": cache_dict["opt_moss_regrets"],
    }
    plot(X, regret_dict, title, xlabel, ylabel, kwargs["plot_var"])
    return (X, regret_dict, title, xlabel, ylabel)


def arm_exp(
    N_EXPS, N_TASKS, HORIZON, OPT_SIZE, N_EXPERT, DS_NAME, n_bandits_list=np.arange(8, 69, 15), quiet=True, **kwargs
):
    cache_dict = _init_cache(N_EXPS, n_bandits_list.shape[0])
    for i in trange(N_EXPS):
        for j, b in enumerate(n_bandits_list):
            kwargs["gap_constrain"] = min(1, np.sqrt(b * np.log(N_TASKS) / HORIZON))
            env = bandit.MetaBernoulli(
                n_bandits=b, opt_size=OPT_SIZE, n_tasks=N_TASKS, n_experts=N_EXPERT, ds_name=DS_NAME, **kwargs
            )
            agent_dict = _init_agents(N_EXPS, N_TASKS, b, HORIZON, OPT_SIZE, N_EXPERT, DS_NAME, env, quiet, **kwargs)
            cache_dict = _collect_data(agent_dict, cache_dict, i, j, N_TASKS, HORIZON, quiet, env, ARM_EXP)
    X = n_bandits_list
    if N_EXPERT is None:
        N_EXPERT = "all"
    title = f"Regret: Horizon {HORIZON}, {N_TASKS} tasks, {N_EXPERT} experts, gap cond. satisfied and subset size {OPT_SIZE}"
    xlabel, ylabel = "Number of Arms", "Regret"
    regret_dict = {
        "moss_regrets": cache_dict["moss_regrets"],
        "EE_regrets": cache_dict["EE_regrets"],
        "PMML_regrets": cache_dict["PMML_regrets"],
        "opt_moss_regrets": cache_dict["opt_moss_regrets"],
    }
    plot(X, regret_dict, title, xlabel, ylabel, kwargs["plot_var"])
    return (X, regret_dict, title, xlabel, ylabel)


def subset_size_exp(N_EXPS, N_TASKS, N_BANDITS, HORIZON, N_EXPERT, DS_NAME, opt_size_list=None, quiet=True, **kwargs):
    if opt_size_list is None:
        opt_size_list = np.arange(1, N_BANDITS + 1, 4)
    cache_dict = _init_cache(N_EXPS, opt_size_list.shape[0])
    for i in trange(N_EXPS):
        for j, s in enumerate(opt_size_list):
            env = bandit.MetaBernoulli(
                n_bandits=N_BANDITS, opt_size=s, n_tasks=N_TASKS, n_experts=N_EXPERT, ds_name=DS_NAME, **kwargs
            )
            agent_dict = _init_agents(N_EXPS, N_TASKS, N_BANDITS, HORIZON, s, N_EXPERT, DS_NAME, env, quiet, **kwargs)
            cache_dict = _collect_data(agent_dict, cache_dict, i, j, N_TASKS, HORIZON, quiet, env, SUBSET_EXP)
    X = opt_size_list
    if N_EXPERT is None:
        N_EXPERT = "all"
    gap = kwargs["gap_constrain"]
    title = f"Regret: {N_BANDITS} arms, Horizon {HORIZON}, {N_TASKS} tasks, gap = {gap:.3f} and {N_EXPERT} experts"
    xlabel, ylabel = "subset size", "Regret"
    regret_dict = {
        "moss_regrets": cache_dict["moss_regrets"],
        "EE_regrets": cache_dict["EE_regrets"],
        "PMML_regrets": cache_dict["PMML_regrets"],
        "opt_moss_regrets": cache_dict["opt_moss_regrets"],
    }
    plot(X, regret_dict, title, xlabel, ylabel, kwargs["plot_var"])
    return (X, regret_dict, title, xlabel, ylabel)
