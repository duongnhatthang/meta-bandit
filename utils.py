import inspect
import algos
import bandit
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from copy import deepcopy
import time
from multiprocessing import Process, Manager

TASK_EXP = 0 # Stochastic
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


def meta_rolls_out(n_tasks, agent, env, horizon, **kwargs):
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
    return regrets


def plot(X, regret_dict, title, xlabel, ylabel, **kwargs):
    # Unfinished runs have regret == -1
    tmp = np.where(regret_dict["opt_moss_regrets"]==-1)[1]
    if tmp.shape[0] == 0:
        opt_moss_max_idx = regret_dict["opt_moss_regrets"].shape[1]
    else:
        opt_moss_max_idx = min(tmp)
    def _get_info(name, max_idx):
        regrets = regret_dict[name+"_regrets"]
        Y = np.mean(regrets, axis=0)
        tmp = np.where(regrets==-1)[1]
        if tmp.shape[0] == 0:
            agent_max_idx = regrets.shape[1]
        else:
            agent_max_idx = min(tmp)
        agent_max_idx = min(max_idx, agent_max_idx)
        dY = 2 * np.sqrt(np.var(regrets, axis=0))
        return regrets, Y, agent_max_idx, dY

    if "moss" not in kwargs['skip_list']:
        moss_regrets, moss_Y, moss_max_idx, moss_dY = _get_info("moss", opt_moss_max_idx)
        plt.errorbar(X[:moss_max_idx], moss_Y[:moss_max_idx], moss_dY[:moss_max_idx], fmt="-", color="#F28522", label="MOSS", linewidth=kwargs['linewidth']) #orange
    if "EE" not in kwargs['skip_list']:
        EE_regrets, EE_Y, EE_max_idx, EE_dY = _get_info("EE", opt_moss_max_idx)
        plt.errorbar(X[:EE_max_idx], EE_Y[:EE_max_idx], EE_dY[:EE_max_idx], fmt="-", color="#009ADE", label="EE", linewidth=kwargs['linewidth']) #blue
    if "PMML" not in kwargs['skip_list']:
        PMML_regrets, PMML_Y, PMML_max_idx, PMML_dY = _get_info("PMML", opt_moss_max_idx)
        plt.errorbar(X[:PMML_max_idx], PMML_Y[:PMML_max_idx], PMML_dY[:PMML_max_idx], fmt="-", color="#00CD6C", label="PMML", linewidth=kwargs['linewidth'])
    if "GML_FC" not in kwargs['skip_list']:
        GML_regrets, GML_FC_Y, GML_FC_max_idx, GML_FC_dY = _get_info("GML_FC", opt_moss_max_idx)
        plt.errorbar(X[:GML_FC_max_idx], GML_FC_Y[:GML_FC_max_idx], GML_FC_dY[:GML_FC_max_idx], fmt="-", color="#AF58BA", label="GML_FC", linewidth=kwargs['linewidth']) #purple
    if "GML" not in kwargs['skip_list']:
        GML_regrets, GML_Y, GML_max_idx, GML_dY = _get_info("GML", opt_moss_max_idx)
        plt.errorbar(X[:GML_max_idx], GML_Y[:GML_max_idx], GML_dY[:GML_max_idx], fmt="-", color="#FF1F5B", label="GML", linewidth=kwargs['linewidth']) #red
    if "OG" not in kwargs['skip_list']:
        OG_regrets, OG_Y, OG_max_idx, OG_dY = _get_info("OG", opt_moss_max_idx)
        plt.errorbar(X[:OG_max_idx], OG_Y[:OG_max_idx], OG_dY[:OG_max_idx], fmt="-", color="#FFC61E", label="OG°", linewidth=kwargs['linewidth']) #yellow
    if "opt_moss" not in kwargs['skip_list']:
        opt_moss_regrets, opt_moss_Y, opt_moss_max_idx, opt_moss_dY = _get_info("opt_moss", opt_moss_max_idx)
        plt.errorbar(X[:opt_moss_max_idx], opt_moss_Y[:opt_moss_max_idx], opt_moss_dY[:opt_moss_max_idx], color="#A0B1BA", label="Opt-MOSS", linestyle="dashed", linewidth=kwargs['linewidth']) #gray

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if kwargs["plot_legend"] is True:
        plt.legend()
    plt.title(title)
    caller_name = inspect.stack()[1][3]
    if kwargs['is_adversarial'] is False:
        setting = "Stochastic"
    else:
        setting = "Adversarial"
    plt.savefig(f"./results/{setting+str(kwargs['n_optimal'])}_{caller_name}.png")


def _init_agents(N_EXPS, N_TASKS, N_ARMS, HORIZON, OPT_SIZE, env, **kwargs):
    output = {}
    if "PMML" not in kwargs['skip_list']:
        PMML_agent = algos.PMML(n_arms=N_ARMS, horizon=HORIZON, n_tasks=N_TASKS, subset_size=OPT_SIZE)
        output["PMML_agent"] = PMML_agent
    if "moss" not in kwargs['skip_list']:
        moss_agent = algos.MOSS(n_arms=N_ARMS, horizon=HORIZON)
        output["moss_agent"] = moss_agent
    if "EE" not in kwargs['skip_list']:
        EE_agent = algos.EE(n_arms=N_ARMS, horizon=HORIZON, n_tasks=N_TASKS, subset_size=OPT_SIZE)
        output["EE_agent"] = EE_agent
    if "opt_moss" not in kwargs['skip_list']:
        opt_moss_agent = algos.ExpertMOSS(n_arms=N_ARMS, horizon=HORIZON, expert_subset=env.opt_indices)
        output["opt_moss_agent"] = opt_moss_agent
    if "GML" not in kwargs['skip_list']:
        GML_agent = algos.GML(n_arms=N_ARMS, horizon=HORIZON, n_tasks=N_TASKS, subset_size=OPT_SIZE)
        output["GML_agent"] = GML_agent
    if "GML_FC" not in kwargs['skip_list']:
        GML_FC_agent = algos.GML_FC(n_arms=N_ARMS, horizon=HORIZON, n_tasks=N_TASKS, subset_size=OPT_SIZE)
        output["GML_FC_agent"] = GML_FC_agent
    if "OG" not in kwargs['skip_list']:
        OG_agent = algos.OG(n_arms=N_ARMS, horizon=HORIZON, n_tasks=N_TASKS, subset_size=OPT_SIZE, **kwargs)
        output["OG_agent"] = OG_agent
    return output


def _init_cache(N_EXPS, x_axis):
    moss_regrets = np.zeros((N_EXPS, x_axis))-1
    EE_regrets = np.zeros((N_EXPS, x_axis))-1
    PMML_regrets = np.zeros((N_EXPS, x_axis))-1
    opt_moss_regrets = np.zeros((N_EXPS, x_axis))-1
    GML_regrets = np.zeros((N_EXPS, x_axis))-1
    GML_FC_regrets = np.zeros((N_EXPS, x_axis))-1
    OG_regrets = np.zeros((N_EXPS, x_axis))-1
    return {
        "moss_regrets": moss_regrets,
        "EE_regrets": EE_regrets,
        "PMML_regrets": PMML_regrets,
        "opt_moss_regrets": opt_moss_regrets,
        "GML_regrets": GML_regrets,
        "GML_FC_regrets": GML_FC_regrets,
        "OG_regrets": OG_regrets,
    }


def _multi_process_wrapper(name, func, return_dict, **kwargs):
    output = func(**kwargs)
    return_dict[name] = output


def _store_collected_data(raw_data_dict, cache_dict, exp_type, i, j, HORIZON, N_TASKS, **kwargs):
    def get_info(name):
        if name not in kwargs['skip_list'] and name+"_r" in raw_data_dict and raw_data_dict[name+"_r"] is not None:
            if exp_type == TASK_EXP:
                cache_dict[name+"_regrets"][i, j] = np.mean(raw_data_dict[name+"_r"]) / N_TASKS
            if exp_type == HORIZON_EXP:
                cache_dict[name+"_regrets"][i, j] = np.mean(raw_data_dict[name+"_r"]) / HORIZON
            else:
                cache_dict[name+"_regrets"][i, j] = np.mean(raw_data_dict[name+"_r"])
        return cache_dict
    cache_dict = get_info("moss")
    cache_dict = get_info("opt_moss")
    cache_dict = get_info("GML")
    cache_dict = get_info("PMML")
    cache_dict = get_info("GML_FC")
    cache_dict = get_info("OG")
    cache_dict = get_info("EE")
    return cache_dict


def _collect_data(agent_dict, cache_dict, i, j, n_tasks, HORIZON, env, exp_type, timer_cache, **kwargs):
    def _rolls_out_and_time(agent, spent_time):
        tic = time.time()
        r = None
        if timer_cache is None or kwargs['timeout']*60 > spent_time:
            r = meta_rolls_out(n_tasks, deepcopy(agent), deepcopy(env), HORIZON, **kwargs)
        else:
            print('Timeout! Ending this task ...')
        toc = time.time()
        return (r, toc - tic)

    return_dict = Manager().dict()

    def _create_process(name):
        tmp_kwargs = {
            "agent":agent_dict[name+"_agent"],
            "spent_time":timer_cache[name]
        }
        return Process(target=_multi_process_wrapper, args=(name, _rolls_out_and_time, return_dict), kwargs=tmp_kwargs)

    if "moss" not in kwargs['skip_list']:
        p_moss = _create_process("moss")
        p_moss.start()
    if "opt_moss" not in kwargs['skip_list']:
        p_opt_moss = _create_process("opt_moss")
        p_opt_moss.start()
    if "EE" not in kwargs['skip_list']:
        p_EE = _create_process("EE")
        p_EE.start()
    if "GML" not in kwargs['skip_list']:
        p_GML = _create_process("GML")
        p_GML.start()
    if "GML_FC" not in kwargs['skip_list']:
        p_GML_FC = _create_process("GML_FC")
        p_GML_FC.start()
    if "OG" not in kwargs['skip_list']:
        p_OG = _create_process("OG")
        p_OG.start()
    if "PMML" not in kwargs['skip_list']:
        p_PMML = _create_process("PMML")
        p_PMML.start()

    raw_data_dict = {}
    if "PMML" not in kwargs['skip_list']:
        p_PMML.join()
        PMML_r = return_dict["PMML"][0]
        timer_cache["PMML"] += return_dict["PMML"][1]
        if PMML_r is not None:
            raw_data_dict["PMML_r"] = PMML_r
    if "moss" not in kwargs['skip_list']:
        p_moss.join()
        moss_r = return_dict["moss"][0]
        timer_cache["moss"] += return_dict["moss"][1]
        if moss_r is not None:
            raw_data_dict["moss_r"] = moss_r
    if "opt_moss" not in kwargs['skip_list']:
        p_opt_moss.join()
        opt_moss_r = return_dict["opt_moss"][0]
        timer_cache["opt_moss"] += return_dict["opt_moss"][1]
        if opt_moss_r is not None:
            raw_data_dict["opt_moss_r"] = opt_moss_r
    if "EE" not in kwargs['skip_list']:
        p_EE.join()
        EE_r = return_dict["EE"][0]
        timer_cache["EE"] += return_dict["EE"][1]
        if EE_r is not None:
            raw_data_dict["EE_r"] = EE_r
    if "GML" not in kwargs['skip_list']:
        p_GML.join()
        GML_r = return_dict["GML"][0]
        timer_cache["GML"] += return_dict["GML"][1]
        if GML_r is not None:
            raw_data_dict["GML_r"] = GML_r
    if "GML_FC" not in kwargs['skip_list']:
        p_GML_FC.join()
        GML_FC_r = return_dict["GML_FC"][0]
        timer_cache["GML_FC"] += return_dict["GML_FC"][1]
        if GML_FC_r is not None:
            raw_data_dict["GML_FC_r"] = GML_FC_r
    if "OG" not in kwargs['skip_list']:
        p_OG.join()
        OG_r = return_dict["OG"][0]
        timer_cache["OG"] += return_dict["OG"][1]
        if OG_r is not None:
            raw_data_dict["OG_r"] = OG_r
    return raw_data_dict, timer_cache


def horizon_exp(
    N_EXPS,
    N_TASKS,
    N_ARMS,
    OPT_SIZE,
    horizon_list=np.arange(1, 202, 50) * 10,
    **kwargs,
):
    cache_dict = _init_cache(N_EXPS, horizon_list.shape[0])
    def _create_process(i):
        tmp_dict = {"rand_seed":np.random.randint(1000)}
        def _sub_routine(rand_seed):
            np.random.seed(rand_seed)
            timer_cache = {'timeout': kwargs['timeout'], "moss":0, "EE":0, "PMML":0, "opt_moss":0, "GML":0, "GML_FC":0, "OG":0,}
            tmp_dict = deepcopy(cache_dict)
            for j, h in enumerate(horizon_list):
                kwargs["gap_constrain"] = min(1, np.sqrt(N_ARMS * np.log(N_TASKS) / h))
                tmp = kwargs["gap_constrain"]
                if kwargs['is_adversarial'] is False:
                    env = bandit.MetaBernoulli(n_arms=N_ARMS, opt_size=OPT_SIZE, n_tasks=N_TASKS, **kwargs)
                else:
                    env = bandit.AdvMetaBernoulli(n_arms=N_ARMS, opt_size=OPT_SIZE, n_tasks=N_TASKS, horizon=h, **kwargs)
                agent_dict = _init_agents(N_EXPS, N_TASKS, N_ARMS, h, OPT_SIZE, env, **kwargs)
                raw_output, timer_cache = _collect_data(agent_dict, cache_dict, i, j, N_TASKS, h, env, HORIZON_EXP, timer_cache, **kwargs)
                tmp_dict = _store_collected_data(raw_output, tmp_dict, HORIZON_EXP, i, j, h, N_TASKS, **kwargs)
            return tmp_dict
        return Process(target=_multi_process_wrapper, args=(i, _sub_routine, return_dict), kwargs=tmp_dict)

    return_dict = Manager().dict()
    processes = []
    for i in trange(N_EXPS):
        p = _create_process(i)
        p.start()
        processes.append(p)

    for i in range(N_EXPS):
        processes[i].join()
        if "moss" not in kwargs['skip_list']:
            cache_dict["moss_regrets"][i] = return_dict[i]["moss_regrets"][i]
        if "EE" not in kwargs['skip_list']:
            cache_dict["EE_regrets"][i] = return_dict[i]["EE_regrets"][i]
        if "opt_moss" not in kwargs['skip_list']:
            cache_dict["opt_moss_regrets"][i] = return_dict[i]["opt_moss_regrets"][i]
        if "GML" not in kwargs['skip_list']:
            cache_dict["GML_regrets"][i] = return_dict[i]["GML_regrets"][i]
        if "GML_FC" not in kwargs['skip_list']:
            cache_dict["GML_FC_regrets"][i] = return_dict[i]["GML_FC_regrets"][i]
        if "OG" not in kwargs['skip_list']:
            cache_dict["OG_regrets"][i] = return_dict[i]["OG_regrets"][i]
        if "PMML" not in kwargs['skip_list']:
            cache_dict["PMML_regrets"][i] = return_dict[i]["PMML_regrets"][i]

    X = horizon_list
    if kwargs['is_adversarial'] is False:
        setting = "Stochastic"
    else:
        setting = "Adversarial"
    title = f"{setting}: {N_ARMS} arms, {N_TASKS} tasks, and subset size = {OPT_SIZE}"
    xlabel, ylabel = "Horizon", "Average Regret per Step"
    plot(X, cache_dict, title, xlabel, ylabel, **kwargs)
    return (X, cache_dict, title, xlabel, ylabel)


def arms_exp(N_EXPS, N_TASKS, HORIZON, OPT_SIZE, n_arms_list=np.arange(8, 69, 15), **kwargs):
    cache_dict = _init_cache(N_EXPS, n_arms_list.shape[0])
    def _create_process(i):
        tmp_dict = {"rand_seed":np.random.randint(1000)}
        def _sub_routine(rand_seed):
            np.random.seed(rand_seed)
            timer_cache = {'timeout': kwargs['timeout'], "moss":0, "EE":0, "PMML":0, "opt_moss":0, "GML":0, "GML_FC":0, "OG":0,}
            tmp_dict = deepcopy(cache_dict)
            for j, b in enumerate(n_arms_list):
                kwargs["gap_constrain"] = min(1, np.sqrt(b * np.log(N_TASKS) / HORIZON))
                if kwargs['is_adversarial'] is False:
                    env = bandit.MetaBernoulli(n_arms=b, opt_size=OPT_SIZE, n_tasks=N_TASKS, **kwargs)
                else:
                    env = bandit.AdvMetaBernoulli(n_arms=b, opt_size=OPT_SIZE, n_tasks=N_TASKS, horizon=HORIZON, **kwargs)
                agent_dict = _init_agents(N_EXPS, N_TASKS, b, HORIZON, OPT_SIZE, env, **kwargs)
                raw_output, timer_cache = _collect_data(agent_dict, cache_dict, i, j, N_TASKS, HORIZON, env, ARM_EXP, timer_cache, **kwargs)
                tmp_dict = _store_collected_data(raw_output, tmp_dict, ARM_EXP, i, j, HORIZON, N_TASKS, **kwargs)
            return tmp_dict
        return Process(target=_multi_process_wrapper, args=(i, _sub_routine, return_dict), kwargs=tmp_dict)

    return_dict = Manager().dict()
    processes = []
    for i in trange(N_EXPS):
        p = _create_process(i)
        p.start()
        processes.append(p)

    for i in range(N_EXPS):
        processes[i].join()
        if "moss" not in kwargs['skip_list']:
            cache_dict["moss_regrets"][i] = return_dict[i]["moss_regrets"][i]
        if "EE" not in kwargs['skip_list']:
            cache_dict["EE_regrets"][i] = return_dict[i]["EE_regrets"][i]
        if "opt_moss" not in kwargs['skip_list']:
            cache_dict["opt_moss_regrets"][i] = return_dict[i]["opt_moss_regrets"][i]
        if "GML" not in kwargs['skip_list']:
            cache_dict["GML_regrets"][i] = return_dict[i]["GML_regrets"][i]
        if "GML_FC" not in kwargs['skip_list']:
            cache_dict["GML_FC_regrets"][i] = return_dict[i]["GML_FC_regrets"][i]
        if "OG" not in kwargs['skip_list']:
            cache_dict["OG_regrets"][i] = return_dict[i]["OG_regrets"][i]
        if "PMML" not in kwargs['skip_list']:
            cache_dict["PMML_regrets"][i] = return_dict[i]["PMML_regrets"][i]

    X = n_arms_list
    if kwargs['is_adversarial'] is False:
        setting = "Stochastic"
    else:
        setting = "Adversarial"
    title = f"{setting}: horizon = {HORIZON}, {N_TASKS} tasks, and subset size = {OPT_SIZE}"
    xlabel, ylabel = "Number of Arms", "Regret"
    plot(X, cache_dict, title, xlabel, ylabel, **kwargs)
    return (X, cache_dict, title, xlabel, ylabel)


def subset_exp(N_EXPS, N_TASKS, N_ARMS, HORIZON, opt_size_list=None, **kwargs):
    if opt_size_list is None:
        opt_size_list = np.arange(1, N_ARMS + 1, 4)
    cache_dict = _init_cache(N_EXPS, opt_size_list.shape[0])
    def _create_process(i):
        tmp_dict = {"rand_seed":np.random.randint(1000)}
        def _sub_routine(rand_seed):
            np.random.seed(rand_seed)
            timer_cache = {'timeout': kwargs['timeout'], "moss":0, "EE":0, "PMML":0, "opt_moss":0, "GML":0, "GML_FC":0, "OG":0,}
            tmp_dict = deepcopy(cache_dict)
            for j, s in enumerate(opt_size_list):
                if kwargs['is_adversarial'] is False:
                    env = bandit.MetaBernoulli(n_arms=N_ARMS, opt_size=s, n_tasks=N_TASKS, **kwargs)
                else:
                    env = bandit.AdvMetaBernoulli(n_arms=N_ARMS, opt_size=s, n_tasks=N_TASKS, horizon=HORIZON, **kwargs)
                agent_dict = _init_agents(N_EXPS, N_TASKS, N_ARMS, HORIZON, s, env, **kwargs)
                raw_output, timer_cache = _collect_data(agent_dict, cache_dict, i, j, N_TASKS, HORIZON, env, SUBSET_EXP, timer_cache, **kwargs)
                tmp_dict = _store_collected_data(raw_output, tmp_dict, SUBSET_EXP, i, j, HORIZON, N_TASKS, **kwargs)
            return tmp_dict
        return Process(target=_multi_process_wrapper, args=(i, _sub_routine, return_dict), kwargs=tmp_dict)

    return_dict = Manager().dict()
    processes = []
    for i in trange(N_EXPS):
        p = _create_process(i)
        p.start()
        processes.append(p)

    for i in range(N_EXPS):
        processes[i].join()
        if "moss" not in kwargs['skip_list']:
            cache_dict["moss_regrets"][i] = return_dict[i]["moss_regrets"][i]
        if "EE" not in kwargs['skip_list']:
            cache_dict["EE_regrets"][i] = return_dict[i]["EE_regrets"][i]
        if "opt_moss" not in kwargs['skip_list']:
            cache_dict["opt_moss_regrets"][i] = return_dict[i]["opt_moss_regrets"][i]
        if "GML" not in kwargs['skip_list']:
            cache_dict["GML_regrets"][i] = return_dict[i]["GML_regrets"][i]
        if "GML_FC" not in kwargs['skip_list']:
            cache_dict["GML_FC_regrets"][i] = return_dict[i]["GML_FC_regrets"][i]
        if "OG" not in kwargs['skip_list']:
            cache_dict["OG_regrets"][i] = return_dict[i]["OG_regrets"][i]
        if "PMML" not in kwargs['skip_list']:
            cache_dict["PMML_regrets"][i] = return_dict[i]["PMML_regrets"][i]

    X = opt_size_list
    gap = kwargs["gap_constrain"]
    if kwargs['is_adversarial'] is False:
        setting = "Stochastic"
    else:
        setting = "Adversarial"
    title = f"{setting}: {N_ARMS} arms, horizon = {HORIZON}, and {N_TASKS} tasks"
    xlabel, ylabel = "subset size", "Regret"
    plot(X, cache_dict, title, xlabel, ylabel, **kwargs)
    return (X, cache_dict, title, xlabel, ylabel)

def task_exp(
    N_EXPS,
    HORIZON,
    N_ARMS,
    OPT_SIZE,
    task_list=np.arange(600, 1002, 100),
    **kwargs,
):
    cache_dict = _init_cache(N_EXPS, task_list.shape[0])
    def _create_process(i):
        tmp_dict = {"rand_seed":np.random.randint(1000)}
        def _sub_routine(rand_seed):
            np.random.seed(rand_seed)
            timer_cache = {'timeout': kwargs['timeout'], "moss":0, "EE":0, "PMML":0, "opt_moss":0, "GML":0, "GML_FC":0, "OG":0,}
            tmp_dict = deepcopy(cache_dict)
            for j, n_t in enumerate(task_list):
                kwargs["gap_constrain"] = min(1, np.sqrt(N_ARMS * np.log(n_t) / HORIZON))
                tmp = kwargs["gap_constrain"]
                if kwargs['is_adversarial'] is False:
                    env = bandit.MetaBernoulli(n_arms=N_ARMS, opt_size=OPT_SIZE, n_tasks=n_t, **kwargs)
                else:
                    env = bandit.AdvMetaBernoulli(n_arms=N_ARMS, opt_size=OPT_SIZE, n_tasks=n_t, horizon=HORIZON, **kwargs)
                agent_dict = _init_agents(N_EXPS, n_t, N_ARMS, HORIZON, OPT_SIZE, env, **kwargs)
                raw_output, timer_cache = _collect_data(agent_dict, cache_dict, i, j, n_t, HORIZON, env, TASK_EXP, timer_cache, **kwargs)
                tmp_dict = _store_collected_data(raw_output, tmp_dict, TASK_EXP, i, j, HORIZON, n_t, **kwargs)
            return tmp_dict
        return Process(target=_multi_process_wrapper, args=(i, _sub_routine, return_dict), kwargs=tmp_dict)

    return_dict = Manager().dict()
    processes = []
    for i in trange(N_EXPS):
        p = _create_process(i)
        p.start()
        processes.append(p)

    for i in range(N_EXPS):
        processes[i].join()
        if "moss" not in kwargs['skip_list']:
            cache_dict["moss_regrets"][i] = return_dict[i]["moss_regrets"][i]
        if "EE" not in kwargs['skip_list']:
            cache_dict["EE_regrets"][i] = return_dict[i]["EE_regrets"][i]
        if "opt_moss" not in kwargs['skip_list']:
            cache_dict["opt_moss_regrets"][i] = return_dict[i]["opt_moss_regrets"][i]
        if "GML" not in kwargs['skip_list']:
            cache_dict["GML_regrets"][i] = return_dict[i]["GML_regrets"][i]
        if "GML_FC" not in kwargs['skip_list']:
            cache_dict["GML_FC_regrets"][i] = return_dict[i]["GML_FC_regrets"][i]
        if "OG" not in kwargs['skip_list']:
            cache_dict["OG_regrets"][i] = return_dict[i]["OG_regrets"][i]
        if "PMML" not in kwargs['skip_list']:
            cache_dict["PMML_regrets"][i] = return_dict[i]["PMML_regrets"][i]

    X = task_list
    if kwargs['is_adversarial'] is False:
        setting = "Stochastic"
    else:
        setting = "Adversarial"
    title = f"{setting}: {N_ARMS} arms, horizon = {HORIZON}, and subset size = {OPT_SIZE}"
    xlabel, ylabel = "Number of tasks", "Average Regret per task"
    plot(X, cache_dict, title, xlabel, ylabel, **kwargs)
    return (X, cache_dict, title, xlabel, ylabel)
