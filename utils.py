import inspect
import time
from copy import deepcopy
from multiprocessing import Manager, Process

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
    EXT_set = []  # For adversarial setting
    for idx in range(n_tasks):
        env.reset_task(idx)
        r = rolls_out(agent, env, horizon, kwargs["quiet"])
        regrets.append(r)
        if kwargs["is_adversarial"] is True:
            if isinstance(agent, algos.ExpertMOSS):  # Optimal MOSS
                EXT_set = env.opt_indices
            elif isinstance(agent, algos.PhaseElim):
                surviving_arms = agent.A_l
                EXT_set = list(set(EXT_set + surviving_arms))
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
    tmp = np.where(regret_dict["opt_moss_regrets"] == -1)[1]
    if tmp.shape[0] == 0:
        opt_moss_max_idx = regret_dict["opt_moss_regrets"].shape[1]
    else:
        opt_moss_max_idx = min(tmp)

    def _get_info(name, max_idx):
        regrets = regret_dict[name + "_regrets"]
        Y = np.mean(regrets, axis=0)
        tmp = np.where(regrets == -1)[1]
        if tmp.shape[0] == 0:
            agent_max_idx = regrets.shape[1]
        else:
            agent_max_idx = min(tmp)
        agent_max_idx = min(max_idx, agent_max_idx)
        dY = 2 * np.sqrt(np.var(regrets, axis=0))
        return regrets, Y, agent_max_idx, dY

    if "moss" not in kwargs["skip_list"]:
        moss_regrets, moss_Y, moss_max_idx, moss_dY = _get_info("moss", opt_moss_max_idx)
        plt.errorbar(
            X[:moss_max_idx],
            moss_Y[:moss_max_idx],
            moss_dY[:moss_max_idx],
            # fmt="-",
            color="#F28522",
            label="MOSS",
            linewidth=kwargs["linewidth"],
        )  # orange
    if "EE" not in kwargs["skip_list"]:
        EE_regrets, EE_Y, EE_max_idx, EE_dY = _get_info("EE", opt_moss_max_idx)
        plt.errorbar(
            X[:EE_max_idx],
            EE_Y[:EE_max_idx],
            EE_dY[:EE_max_idx],
            # fmt="-",
            color="#009ADE",
            label="EE",
            linewidth=kwargs["linewidth"],
        )  # blue
    if "E_BASS" not in kwargs["skip_list"]:
        E_BASS_regrets, E_BASS_Y, E_BASS_max_idx, E_BASS_dY = _get_info("E_BASS", opt_moss_max_idx)
        plt.errorbar(
            X[:E_BASS_max_idx],
            E_BASS_Y[:E_BASS_max_idx],
            E_BASS_dY[:E_BASS_max_idx],
            # fmt="-",
            color="#00CD6C",
            label="E-BASS",
            linewidth=kwargs["linewidth"],
            linestyle=(0, (3,1,1,1,1, 1)),
        )
    if "G_BASS_FC" not in kwargs["skip_list"]:
        G_BASS_regrets, G_BASS_FC_Y, G_BASS_FC_max_idx, G_BASS_FC_dY = _get_info("G_BASS_FC", opt_moss_max_idx)
        plt.errorbar(
            X[:G_BASS_FC_max_idx],
            G_BASS_FC_Y[:G_BASS_FC_max_idx],
            G_BASS_FC_dY[:G_BASS_FC_max_idx],
            # fmt="-",
            color="#AF58BA",
            label="G-BASS-FC",
            linewidth=kwargs["linewidth"],
        )  # purple
    if "G_BASS" not in kwargs["skip_list"]:
        G_BASS_regrets, G_BASS_Y, G_BASS_max_idx, G_BASS_dY = _get_info("G_BASS", opt_moss_max_idx)
        plt.errorbar(
            X[:G_BASS_max_idx],
            G_BASS_Y[:G_BASS_max_idx],
            G_BASS_dY[:G_BASS_max_idx],
            # fmt="-",
            color="#FF1F5B",
            label="G-BASS",
            linewidth=kwargs["linewidth"],
            linestyle=(0, (5, 1)),
        )  # red
    if "OG" not in kwargs["skip_list"]:
        OG_regrets, OG_Y, OG_max_idx, OG_dY = _get_info("OG", opt_moss_max_idx)
        plt.errorbar(
            X[:OG_max_idx],
            OG_Y[:OG_max_idx],
            OG_dY[:OG_max_idx],
            # fmt="-",
            color="#FFC61E",
            label="OG°",
            linewidth=kwargs["linewidth"],
            linestyle="-.",
        )  # yellow
    if "OS_BASS" not in kwargs["skip_list"]:
        OS_BASS_regrets, OS_BASS_Y, OS_BASS_max_idx, OS_BASS_dY = _get_info("OS_BASS", opt_moss_max_idx)
        plt.errorbar(
            X[:OS_BASS_max_idx],
            OS_BASS_Y[:OS_BASS_max_idx],
            OS_BASS_dY[:OS_BASS_max_idx],
            # fmt="-",
            color="#AEEA00",
            label="OS-BASS",
            linewidth=kwargs["linewidth"],
            linestyle="dotted",
        )  # lime
    if "opt_moss" not in kwargs["skip_list"]:
        opt_moss_regrets, opt_moss_Y, opt_moss_max_idx, opt_moss_dY = _get_info("opt_moss", opt_moss_max_idx)
        plt.errorbar(
            X[:opt_moss_max_idx],
            opt_moss_Y[:opt_moss_max_idx],
            opt_moss_dY[:opt_moss_max_idx],
            color="#A0B1BA",
            label="Opt-MOSS",
            linestyle="dashed",
            linewidth=kwargs["linewidth"],
        )  # gray

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if kwargs["plot_legend"] is True:
        plt.legend()
    plt.title(title)
    caller_name = inspect.stack()[1][3]
    if kwargs["is_adversarial"] is False:
        setting = "Stochastic"
    else:
        setting = "Adversarial"
    if caller_name != "<module>":
        plt.savefig(f"./results/{setting}_{caller_name}_{str(time.time())}.png")


def _init_agents(N_EXPS, N_TASKS, N_ARMS, HORIZON, OPT_SIZE, env, **kwargs):
    output = {}
    if "E_BASS" not in kwargs["skip_list"]:
        E_BASS_agent = algos.E_BASS(n_arms=N_ARMS, horizon=HORIZON, n_tasks=N_TASKS, subset_size=OPT_SIZE)
        output["E_BASS_agent"] = E_BASS_agent
    if "moss" not in kwargs["skip_list"]:
        moss_agent = algos.MOSS(n_arms=N_ARMS, horizon=HORIZON)
        output["moss_agent"] = moss_agent
    if "EE" not in kwargs["skip_list"]:
        EE_agent = algos.EE(n_arms=N_ARMS, horizon=HORIZON, n_tasks=N_TASKS, subset_size=OPT_SIZE)
        output["EE_agent"] = EE_agent
    if "opt_moss" not in kwargs["skip_list"]:
        opt_moss_agent = algos.ExpertMOSS(n_arms=N_ARMS, horizon=HORIZON, expert_subset=env.opt_indices)
        output["opt_moss_agent"] = opt_moss_agent
    if "G_BASS" not in kwargs["skip_list"]:
        G_BASS_agent = algos.G_BASS(n_arms=N_ARMS, horizon=HORIZON, n_tasks=N_TASKS, subset_size=OPT_SIZE)
        output["G_BASS_agent"] = G_BASS_agent
    if "G_BASS_FC" not in kwargs["skip_list"]:
        G_BASS_FC_agent = algos.G_BASS_FC(n_arms=N_ARMS, horizon=HORIZON, n_tasks=N_TASKS, subset_size=OPT_SIZE)
        output["G_BASS_FC_agent"] = G_BASS_FC_agent
    if "OG" not in kwargs["skip_list"]:
        OG_agent = algos.OG(n_arms=N_ARMS, horizon=HORIZON, n_tasks=N_TASKS, subset_size=OPT_SIZE, **kwargs)
        output["OG_agent"] = OG_agent
    if "OS_BASS" not in kwargs["skip_list"]:
        OS_BASS_agent = algos.OS_BASS(n_arms=N_ARMS, horizon=HORIZON, n_tasks=N_TASKS, subset_size=OPT_SIZE, **kwargs)
        output["OS_BASS_agent"] = OS_BASS_agent
    return output


def _init_cache(N_EXPS, x_axis):
    moss_regrets = np.zeros((N_EXPS, x_axis)) - 1
    EE_regrets = np.zeros((N_EXPS, x_axis)) - 1
    E_BASS_regrets = np.zeros((N_EXPS, x_axis)) - 1
    opt_moss_regrets = np.zeros((N_EXPS, x_axis)) - 1
    G_BASS_regrets = np.zeros((N_EXPS, x_axis)) - 1
    G_BASS_FC_regrets = np.zeros((N_EXPS, x_axis)) - 1
    OG_regrets = np.zeros((N_EXPS, x_axis)) - 1
    OS_BASS_regrets = np.zeros((N_EXPS, x_axis)) - 1
    return {
        "moss_regrets": moss_regrets,
        "EE_regrets": EE_regrets,
        "E_BASS_regrets": E_BASS_regrets,
        "opt_moss_regrets": opt_moss_regrets,
        "G_BASS_regrets": G_BASS_regrets,
        "G_BASS_FC_regrets": G_BASS_FC_regrets,
        "OG_regrets": OG_regrets,
        "OS_BASS_regrets": OS_BASS_regrets,
    }


def _multi_process_wrapper(name, func, return_dict, **kwargs):
    output = func(**kwargs)
    return_dict[name] = output


def _store_collected_data(raw_data_dict, cache_dict, exp_type, i, j, HORIZON, N_TASKS, **kwargs):
    def get_info(name):
        if name not in kwargs["skip_list"] and name + "_r" in raw_data_dict and raw_data_dict[name + "_r"] is not None:
            if exp_type == TASK_EXP:
                cache_dict[name + "_regrets"][i, j] = np.mean(raw_data_dict[name + "_r"]) / N_TASKS
            if exp_type == HORIZON_EXP:
                cache_dict[name + "_regrets"][i, j] = np.mean(raw_data_dict[name + "_r"]) / HORIZON
            else:
                cache_dict[name + "_regrets"][i, j] = np.mean(raw_data_dict[name + "_r"])
        return cache_dict

    cache_dict = get_info("moss")
    cache_dict = get_info("opt_moss")
    cache_dict = get_info("G_BASS")
    cache_dict = get_info("E_BASS")
    cache_dict = get_info("G_BASS_FC")
    cache_dict = get_info("OG")
    cache_dict = get_info("OS_BASS")
    cache_dict = get_info("EE")
    return cache_dict


def _collect_data(agent_dict, cache_dict, i, j, n_tasks, HORIZON, env, exp_type, timer_cache, **kwargs):
    def _rolls_out_and_time(agent, spent_time):
        tic = time.time()
        r = None
        if timer_cache is None or kwargs["timeout"] * 60 > spent_time:
            r = meta_rolls_out(n_tasks, deepcopy(agent), deepcopy(env), HORIZON, **kwargs)
        else:
            print("Timeout! Ending this task ...")
        toc = time.time()
        return (r, toc - tic)

    return_dict = Manager().dict()

    def _create_process(name):
        tmp_kwargs = {"agent": agent_dict[name + "_agent"], "spent_time": timer_cache[name]}
        return Process(target=_multi_process_wrapper, args=(name, _rolls_out_and_time, return_dict), kwargs=tmp_kwargs)

    if "moss" not in kwargs["skip_list"]:
        p_moss = _create_process("moss")
        p_moss.start()
    if "opt_moss" not in kwargs["skip_list"]:
        p_opt_moss = _create_process("opt_moss")
        p_opt_moss.start()
    if "EE" not in kwargs["skip_list"]:
        p_EE = _create_process("EE")
        p_EE.start()
    if "G_BASS" not in kwargs["skip_list"]:
        p_G_BASS = _create_process("G_BASS")
        p_G_BASS.start()
    if "G_BASS_FC" not in kwargs["skip_list"]:
        p_G_BASS_FC = _create_process("G_BASS_FC")
        p_G_BASS_FC.start()
    if "OG" not in kwargs["skip_list"]:
        p_OG = _create_process("OG")
        p_OG.start()
    if "OS_BASS" not in kwargs["skip_list"]:
        p_OS_BASS = _create_process("OS_BASS")
        p_OS_BASS.start()
    if "E_BASS" not in kwargs["skip_list"]:
        p_E_BASS = _create_process("E_BASS")
        p_E_BASS.start()

    raw_data_dict = {}
    if "E_BASS" not in kwargs["skip_list"]:
        p_E_BASS.join()
        E_BASS_r = return_dict["E_BASS"][0]
        timer_cache["E_BASS"] += return_dict["E_BASS"][1]
        if E_BASS_r is not None:
            raw_data_dict["E_BASS_r"] = E_BASS_r
    if "moss" not in kwargs["skip_list"]:
        p_moss.join()
        moss_r = return_dict["moss"][0]
        timer_cache["moss"] += return_dict["moss"][1]
        if moss_r is not None:
            raw_data_dict["moss_r"] = moss_r
    if "opt_moss" not in kwargs["skip_list"]:
        p_opt_moss.join()
        opt_moss_r = return_dict["opt_moss"][0]
        timer_cache["opt_moss"] += return_dict["opt_moss"][1]
        if opt_moss_r is not None:
            raw_data_dict["opt_moss_r"] = opt_moss_r
    if "EE" not in kwargs["skip_list"]:
        p_EE.join()
        EE_r = return_dict["EE"][0]
        timer_cache["EE"] += return_dict["EE"][1]
        if EE_r is not None:
            raw_data_dict["EE_r"] = EE_r
    if "G_BASS" not in kwargs["skip_list"]:
        p_G_BASS.join()
        G_BASS_r = return_dict["G_BASS"][0]
        timer_cache["G_BASS"] += return_dict["G_BASS"][1]
        if G_BASS_r is not None:
            raw_data_dict["G_BASS_r"] = G_BASS_r
    if "G_BASS_FC" not in kwargs["skip_list"]:
        p_G_BASS_FC.join()
        G_BASS_FC_r = return_dict["G_BASS_FC"][0]
        timer_cache["G_BASS_FC"] += return_dict["G_BASS_FC"][1]
        if G_BASS_FC_r is not None:
            raw_data_dict["G_BASS_FC_r"] = G_BASS_FC_r
    if "OG" not in kwargs["skip_list"]:
        p_OG.join()
        OG_r = return_dict["OG"][0]
        timer_cache["OG"] += return_dict["OG"][1]
        if OG_r is not None:
            raw_data_dict["OG_r"] = OG_r
    if "OS_BASS" not in kwargs["skip_list"]:
        p_OS_BASS.join()
        OS_BASS_r = return_dict["OS_BASS"][0]
        timer_cache["OS_BASS"] += return_dict["OS_BASS"][1]
        if OS_BASS_r is not None:
            raw_data_dict["OS_BASS_r"] = OS_BASS_r
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
        tmp_dict = {"rand_seed": np.random.randint(1000)}

        def _sub_routine(rand_seed):
            np.random.seed(rand_seed)
            timer_cache = {
                "timeout": kwargs["timeout"],
                "moss": 0,
                "EE": 0,
                "E_BASS": 0,
                "opt_moss": 0,
                "G_BASS": 0,
                "G_BASS_FC": 0,
                "OG": 0,
                "OS_BASS": 0,
            }
            tmp_dict = deepcopy(cache_dict)
            for j, h in enumerate(horizon_list):
                verify_params(N_TASKS, N_ARMS, h, OPT_SIZE, **kwargs)
                if kwargs["gap_constrain"] is not None:
                    kwargs["gap_constrain"] = min(1, np.sqrt(N_ARMS * np.log(N_TASKS) / h))
                if kwargs["is_adversarial"] is False:
                    env = bandit.MetaStochastic(n_arms=N_ARMS, opt_size=OPT_SIZE, n_tasks=N_TASKS, horizon=h, **kwargs)
                elif kwargs["is_non_oblivious"] is True:
                    env = bandit.NonObliviousMetaAdversarial(
                        n_arms=N_ARMS, opt_size=OPT_SIZE, n_tasks=N_TASKS, horizon=h, **kwargs
                    )
                else:
                    env = bandit.MetaAdversarial(
                        n_arms=N_ARMS, opt_size=OPT_SIZE, n_tasks=N_TASKS, horizon=h, **kwargs
                    )
                agent_dict = _init_agents(N_EXPS, N_TASKS, N_ARMS, h, OPT_SIZE, env, **kwargs)
                raw_output, timer_cache = _collect_data(
                    agent_dict, cache_dict, i, j, N_TASKS, h, env, HORIZON_EXP, timer_cache, **kwargs
                )
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
        if "moss" not in kwargs["skip_list"]:
            cache_dict["moss_regrets"][i] = return_dict[i]["moss_regrets"][i]
        if "EE" not in kwargs["skip_list"]:
            cache_dict["EE_regrets"][i] = return_dict[i]["EE_regrets"][i]
        if "opt_moss" not in kwargs["skip_list"]:
            cache_dict["opt_moss_regrets"][i] = return_dict[i]["opt_moss_regrets"][i]
        if "G_BASS" not in kwargs["skip_list"]:
            cache_dict["G_BASS_regrets"][i] = return_dict[i]["G_BASS_regrets"][i]
        if "G_BASS_FC" not in kwargs["skip_list"]:
            cache_dict["G_BASS_FC_regrets"][i] = return_dict[i]["G_BASS_FC_regrets"][i]
        if "OG" not in kwargs["skip_list"]:
            cache_dict["OG_regrets"][i] = return_dict[i]["OG_regrets"][i]
        if "OS_BASS" not in kwargs["skip_list"]:
            cache_dict["OS_BASS_regrets"][i] = return_dict[i]["OS_BASS_regrets"][i]
        if "E_BASS" not in kwargs["skip_list"]:
            cache_dict["E_BASS_regrets"][i] = return_dict[i]["E_BASS_regrets"][i]

    X = horizon_list
    if kwargs["is_adversarial"] is False:
        setting = "Stochastic"
    else:
        setting = "Adversarial"
    title = f"{setting}: {N_ARMS} arms, {N_TASKS} tasks, and subset size = {OPT_SIZE}"
    xlabel, ylabel = "Horizon (T)", "Average Regret per Step"
    plot(X, cache_dict, title, xlabel, ylabel, **kwargs)
    return (X, cache_dict, title, xlabel, ylabel)


def arms_exp(N_EXPS, N_TASKS, OPT_SIZE, HORIZON, n_arms_list=np.arange(8, 69, 15), **kwargs):
    cache_dict = _init_cache(N_EXPS, n_arms_list.shape[0])

    def _create_process(i):
        tmp_dict = {"rand_seed": np.random.randint(1000)}

        def _sub_routine(rand_seed):
            np.random.seed(rand_seed)
            timer_cache = {
                "timeout": kwargs["timeout"],
                "moss": 0,
                "EE": 0,
                "E_BASS": 0,
                "opt_moss": 0,
                "G_BASS": 0,
                "G_BASS_FC": 0,
                "OG": 0,
                "OS_BASS": 0,
            }
            tmp_dict = deepcopy(cache_dict)
            for j, b in enumerate(n_arms_list):
                verify_params(N_TASKS, b, HORIZON, OPT_SIZE, **kwargs)
                if kwargs["gap_constrain"] is not None:
                    kwargs["gap_constrain"] = min(1, np.sqrt(b * np.log(N_TASKS) / HORIZON))
                if kwargs["is_adversarial"] is False:
                    env = bandit.MetaStochastic(n_arms=b, opt_size=OPT_SIZE, n_tasks=N_TASKS, horizon=HORIZON, **kwargs)
                elif kwargs["is_non_oblivious"] is True:
                    env = bandit.NonObliviousMetaAdversarial(
                        n_arms=b, opt_size=OPT_SIZE, n_tasks=N_TASKS, horizon=HORIZON, **kwargs
                    )
                else:
                    env = bandit.MetaAdversarial(
                        n_arms=b, opt_size=OPT_SIZE, n_tasks=N_TASKS, horizon=HORIZON, **kwargs
                    )
                agent_dict = _init_agents(N_EXPS, N_TASKS, b, HORIZON, OPT_SIZE, env, **kwargs)
                raw_output, timer_cache = _collect_data(
                    agent_dict, cache_dict, i, j, N_TASKS, HORIZON, env, ARM_EXP, timer_cache, **kwargs
                )
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
        if "moss" not in kwargs["skip_list"]:
            cache_dict["moss_regrets"][i] = return_dict[i]["moss_regrets"][i]
        if "EE" not in kwargs["skip_list"]:
            cache_dict["EE_regrets"][i] = return_dict[i]["EE_regrets"][i]
        if "opt_moss" not in kwargs["skip_list"]:
            cache_dict["opt_moss_regrets"][i] = return_dict[i]["opt_moss_regrets"][i]
        if "G_BASS" not in kwargs["skip_list"]:
            cache_dict["G_BASS_regrets"][i] = return_dict[i]["G_BASS_regrets"][i]
        if "G_BASS_FC" not in kwargs["skip_list"]:
            cache_dict["G_BASS_FC_regrets"][i] = return_dict[i]["G_BASS_FC_regrets"][i]
        if "OG" not in kwargs["skip_list"]:
            cache_dict["OG_regrets"][i] = return_dict[i]["OG_regrets"][i]
        if "OS_BASS" not in kwargs["skip_list"]:
            cache_dict["OS_BASS_regrets"][i] = return_dict[i]["OS_BASS_regrets"][i]
        if "E_BASS" not in kwargs["skip_list"]:
            cache_dict["E_BASS_regrets"][i] = return_dict[i]["E_BASS_regrets"][i]

    X = n_arms_list
    if kwargs["is_adversarial"] is False:
        setting = "Stochastic"
    else:
        setting = "Adversarial"
    title = f"{setting}: horizon = {HORIZON}, {N_TASKS} tasks, and subset size = {OPT_SIZE}"
    xlabel, ylabel = "Number of Arms (K)", "Regret"
    plot(X, cache_dict, title, xlabel, ylabel, **kwargs)
    return (X, cache_dict, title, xlabel, ylabel)


def subset_exp(N_EXPS, N_TASKS, N_ARMS, HORIZON, opt_size_list=None, **kwargs):
    if opt_size_list is None:
        opt_size_list = np.arange(1, N_ARMS + 1, 4)
    cache_dict = _init_cache(N_EXPS, opt_size_list.shape[0])

    def _create_process(i):
        tmp_dict = {"rand_seed": np.random.randint(1000)}

        def _sub_routine(rand_seed):
            np.random.seed(rand_seed)
            timer_cache = {
                "timeout": kwargs["timeout"],
                "moss": 0,
                "EE": 0,
                "E_BASS": 0,
                "opt_moss": 0,
                "G_BASS": 0,
                "G_BASS_FC": 0,
                "OG": 0,
                "OS_BASS": 0,
            }
            tmp_dict = deepcopy(cache_dict)
            for j, s in enumerate(opt_size_list):
                verify_params(N_TASKS, N_ARMS, HORIZON, s, **kwargs)
                if kwargs["is_adversarial"] is False:
                    env = bandit.MetaStochastic(n_arms=N_ARMS, opt_size=s, n_tasks=N_TASKS, horizon=HORIZON, **kwargs)
                elif kwargs["is_non_oblivious"] is True:
                    env = bandit.NonObliviousMetaAdversarial(
                        n_arms=N_ARMS, opt_size=s, n_tasks=N_TASKS, horizon=HORIZON, **kwargs
                    )
                else:
                    env = bandit.MetaAdversarial(n_arms=N_ARMS, opt_size=s, n_tasks=N_TASKS, horizon=HORIZON, **kwargs)
                agent_dict = _init_agents(N_EXPS, N_TASKS, N_ARMS, HORIZON, s, env, **kwargs)
                raw_output, timer_cache = _collect_data(
                    agent_dict, cache_dict, i, j, N_TASKS, HORIZON, env, SUBSET_EXP, timer_cache, **kwargs
                )
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
        if "moss" not in kwargs["skip_list"]:
            cache_dict["moss_regrets"][i] = return_dict[i]["moss_regrets"][i]
        if "EE" not in kwargs["skip_list"]:
            cache_dict["EE_regrets"][i] = return_dict[i]["EE_regrets"][i]
        if "opt_moss" not in kwargs["skip_list"]:
            cache_dict["opt_moss_regrets"][i] = return_dict[i]["opt_moss_regrets"][i]
        if "G_BASS" not in kwargs["skip_list"]:
            cache_dict["G_BASS_regrets"][i] = return_dict[i]["G_BASS_regrets"][i]
        if "G_BASS_FC" not in kwargs["skip_list"]:
            cache_dict["G_BASS_FC_regrets"][i] = return_dict[i]["G_BASS_FC_regrets"][i]
        if "OG" not in kwargs["skip_list"]:
            cache_dict["OG_regrets"][i] = return_dict[i]["OG_regrets"][i]
        if "OS_BASS" not in kwargs["skip_list"]:
            cache_dict["OS_BASS_regrets"][i] = return_dict[i]["OS_BASS_regrets"][i]
        if "E_BASS" not in kwargs["skip_list"]:
            cache_dict["E_BASS_regrets"][i] = return_dict[i]["E_BASS_regrets"][i]

    X = opt_size_list
    if kwargs["is_adversarial"] is False:
        setting = "Stochastic"
    else:
        setting = "Adversarial"
    title = f"{setting}: {N_ARMS} arms, horizon = {HORIZON}, and {N_TASKS} tasks"
    xlabel, ylabel = "Subset size (M)", "Regret"
    plot(X, cache_dict, title, xlabel, ylabel, **kwargs)
    return (X, cache_dict, title, xlabel, ylabel)


def task_exp(
    N_EXPS,
    N_ARMS,
    OPT_SIZE,
    HORIZON,
    task_list=np.arange(600, 1002, 100),
    **kwargs,
):
    cache_dict = _init_cache(N_EXPS, task_list.shape[0])

    def _create_process(i):
        tmp_dict = {"rand_seed": np.random.randint(1000)}

        def _sub_routine(rand_seed):
            np.random.seed(rand_seed)
            timer_cache = {
                "timeout": kwargs["timeout"],
                "moss": 0,
                "EE": 0,
                "E_BASS": 0,
                "opt_moss": 0,
                "G_BASS": 0,
                "G_BASS_FC": 0,
                "OG": 0,
                "OS_BASS": 0,
            }
            tmp_dict = deepcopy(cache_dict)
            for j, n_t in enumerate(task_list):
                verify_params(n_t, N_ARMS, HORIZON, OPT_SIZE, **kwargs)
                if kwargs["gap_constrain"] is not None:
                    kwargs["gap_constrain"] = min(1, np.sqrt(N_ARMS * np.log(n_t) / HORIZON))
                if kwargs["is_adversarial"] is False:
                    env = bandit.MetaStochastic(n_arms=N_ARMS, opt_size=OPT_SIZE, n_tasks=n_t, horizon=HORIZON, **kwargs)
                elif kwargs["is_non_oblivious"] is True:
                    env = bandit.NonObliviousMetaAdversarial(
                        n_arms=N_ARMS, opt_size=OPT_SIZE, n_tasks=n_t, horizon=HORIZON, **kwargs
                    )
                else:
                    env = bandit.MetaAdversarial(
                        n_arms=N_ARMS, opt_size=OPT_SIZE, n_tasks=n_t, horizon=HORIZON, **kwargs
                    )
                agent_dict = _init_agents(N_EXPS, n_t, N_ARMS, HORIZON, OPT_SIZE, env, **kwargs)
                raw_output, timer_cache = _collect_data(
                    agent_dict, cache_dict, i, j, n_t, HORIZON, env, TASK_EXP, timer_cache, **kwargs
                )
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
        if "moss" not in kwargs["skip_list"]:
            cache_dict["moss_regrets"][i] = return_dict[i]["moss_regrets"][i]
        if "EE" not in kwargs["skip_list"]:
            cache_dict["EE_regrets"][i] = return_dict[i]["EE_regrets"][i]
        if "opt_moss" not in kwargs["skip_list"]:
            cache_dict["opt_moss_regrets"][i] = return_dict[i]["opt_moss_regrets"][i]
        if "G_BASS" not in kwargs["skip_list"]:
            cache_dict["G_BASS_regrets"][i] = return_dict[i]["G_BASS_regrets"][i]
        if "G_BASS_FC" not in kwargs["skip_list"]:
            cache_dict["G_BASS_FC_regrets"][i] = return_dict[i]["G_BASS_FC_regrets"][i]
        if "OG" not in kwargs["skip_list"]:
            cache_dict["OG_regrets"][i] = return_dict[i]["OG_regrets"][i]
        if "OS_BASS" not in kwargs["skip_list"]:
            cache_dict["OS_BASS_regrets"][i] = return_dict[i]["OS_BASS_regrets"][i]
        if "E_BASS" not in kwargs["skip_list"]:
            cache_dict["E_BASS_regrets"][i] = return_dict[i]["E_BASS_regrets"][i]

    X = task_list
    if kwargs["is_adversarial"] is False:
        setting = "Stochastic"
    else:
        setting = "Adversarial"
    title = f"{setting}: {N_ARMS} arms, horizon = {HORIZON}, and subset size = {OPT_SIZE}"
    xlabel, ylabel = "Number of tasks (T)", "Average Regret per task"
    plot(X, cache_dict, title, xlabel, ylabel, **kwargs)
    return (X, cache_dict, title, xlabel, ylabel)

def verify_params(n_tasks, n_arms, tau, subset_size, **kwargs):
    assert n_arms<=tau, f"The number of arm ({n_arms}) must be smaller than the horizon ({tau})"
    assert subset_size<=n_arms and subset_size>1, f"The subset size ({subset_size}) must be smaller than the number of arm ({n_arms}) and >1"
    m_i = 16*np.log(n_tasks)
    if n_arms*m_i > tau:
        print(f"verify_params WARNING (Phased Elimination): phase 1 duration ({n_arms*m_i}) is larger than the horizon ({tau}) \n=> increase horizon, decrease n_arms or/and n_tasks.")
    if "OG" not in kwargs['skip_list']:
        og_gamma = kwargs['OG_scale']*subset_size*(1+np.log(n_tasks))*(n_arms*np.log(n_arms)/n_tasks)**(1/3)
        if og_gamma<0 or og_gamma>1:
            print(f"WARNING (OG baseline): og_gamma ({og_gamma}) must in range [0,1]. Capped at 1.")
