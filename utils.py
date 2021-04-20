import numpy as np
import matplotlib.pyplot as plt
import algos
import bandit
from tqdm import tqdm, trange

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
    if quiet == False:
        for i in trange(horizon):
            a = agent.get_action(obs)
            next_obs, r, _, _ = env.step(a)
            rewards.append(r)
            obs = next_obs
            if hasattr(agent, 'update'):
                agent.update(a, r)
    else:
        for i in range(horizon):
            a = agent.get_action(obs)
            next_obs, r, _, _ = env.step(a)
            rewards.append(r)
            obs = next_obs
            if hasattr(agent, 'update'):
                agent.update(a, r)
    if hasattr(agent, 'eps_end_update'):
        agent.eps_end_update(obs)
    if isinstance(agent, algos.Exp3) == True and agent.is_reset == True:
        agent.reset()
    regret = np.max(env._p)*horizon - np.sum(rewards)
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
        regrets.append(np.average(tmp_regrets)) #average regret until this task
    return regrets

def plot(X, regret_dict, title, xlabel, ylabel, plot_var = False):
    moss_regrets = regret_dict['moss_regrets']
    EE_regrets = regret_dict['EE_regrets']
    PMML_regrets = regret_dict['PMML_regrets']
    opt_moss_regrets = regret_dict['opt_moss_regrets']
#     EWAmaxStats_regrets = regret_dict['EWAmaxStats_regrets']
#     exp3_regrets = regret_dict['exp3_regrets']
#     exp3_reset_regrets = regret_dict['exp3_reset_regrets']
#     EWAmaxStatsTrick_regrets = regret_dict['EWAmaxStatsTrick_regrets']
#     PMML_EWA_regrets = regret_dict['PMML_EWA_regrets']

    moss_Y = np.mean(moss_regrets, axis=0)
    EE_Y = np.mean(EE_regrets, axis=0)
    PMML_Y = np.mean(PMML_regrets, axis=0)
    opt_moss_Y = np.mean(opt_moss_regrets, axis=0)
#     EWAmaxStats_Y = np.mean(EWAmaxStats_regrets, axis=0)
#     EWAmaxStatsTrick_Y = np.mean(EWAmaxStatsTrick_regrets, axis=0)
#     exp3_Y = np.mean(exp3_regrets, axis=0)
#     exp3_reset_Y = np.mean(exp3_reset_regrets, axis=0)
#     PMML_EWA_Y = np.mean(PMML_EWA_regrets, axis=0)
    if plot_var == True:
        moss_dY = 2*np.sqrt(np.var(moss_regrets, axis=0))
        EE_dY = 2*np.sqrt(np.var(EE_regrets, axis=0))
        PMML_dY = 2*np.sqrt(np.var(PMML_regrets, axis=0))
        opt_moss_dY = 2*np.sqrt(np.var(opt_moss_regrets, axis=0))
#         EWAmaxStats_dY = 2*np.sqrt(np.var(EWAmaxStats_regrets, axis=0))
#         EWAmaxStatsTrick_dY = 2*np.sqrt(np.var(EWAmaxStatsTrick_regrets, axis=0))
#         exp3_dY = 2*np.sqrt(np.var(exp3_regrets, axis=0))
#         exp3_reset_dY = 2*np.sqrt(np.var(exp3_reset_regrets, axis=0))
#         PMML_EWA_dY = 2*np.sqrt(np.var(PMML_EWA_regrets, axis=0))

        plt.errorbar(X, moss_Y, moss_dY, fmt='-', color='green', label = "MOSS")
        plt.errorbar(X, EE_Y, EE_dY, fmt='-', color='blue', label = "EE")
        plt.errorbar(X, PMML_Y, PMML_dY, fmt='-', color='red', label = "PMML")
        plt.errorbar(X, opt_moss_Y, opt_moss_dY, fmt='-', color='black', label = "Optimal MOSS")
    else:
        plt.plot(X, moss_Y, '-', color='green', label = "MOSS")
        plt.plot(X, EE_Y, '-', color='blue', label = "EE")
        plt.plot(X, PMML_Y, '-', color='red', label = "PMML")
        plt.plot(X, opt_moss_Y, '-', color='black', label = "Optimal MOSS")
    #     plt.plot(X, EWAmaxStats_Y, '-', color='orange', label = "EWA")
    #     plt.plot(X, EWAmaxStatsTrick_Y, '-', color='purple', label = "EWA (all data)")
    #     plt.plot(X, exp3_Y, '-', color='tomato', label = "EXP3")
    #     plt.plot(X, exp3_reset_Y, '-', color='yellow', label = "EXP3 Reset")
    #     plt.plot(X, PMML_EWA_Y, '-', color='cyan', label = "PMML")
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)

def _init_agents(N_EXPS, N_TASKS, N_BANDITS, HORIZON, OPT_SIZE, N_EXPERT, DS_NAME, env, quiet=True, **kwargs):
    moss_agent = algos.MOSS(n_bandits=N_BANDITS, horizon=HORIZON)
    EE_agent = algos.EE(n_bandits=N_BANDITS, horizon=HORIZON, 
                            n_tasks=N_TASKS, expert_subsets=env.expert_subsets)
    PMML_agent = algos.PMML(n_bandits=N_BANDITS, horizon=HORIZON, n_tasks=N_TASKS,
                      expert_subsets=env.expert_subsets)
    opt_moss_agent = algos.ExpertMOSS(n_bandits=N_BANDITS, horizon=HORIZON, expert_subset=env.opt_indices)
#     exp3_kwargs = {'n_tasks':N_TASKS}
#     exp3_agent = algos.Exp3(n_bandits=N_BANDITS, horizon=HORIZON, **exp3_kwargs)
#     exp3_reset_agent = algos.Exp3(n_bandits=N_BANDITS, horizon=HORIZON, is_reset=True)
#     EWAmaxStats_agent = algos.EWAmaxStats(n_bandits=N_BANDITS, horizon=HORIZON, n_tasks=N_TASKS, 
#                       n_unbiased_obs=kwargs['unbiased_obs'], 
#                       expert_subsets=env.expert_subsets)
#     EWAmaxStatsTrick_agent = algos.EWAmaxStats(n_bandits=N_BANDITS, horizon=HORIZON, n_tasks=N_TASKS, 
#                       n_unbiased_obs=kwargs['unbiased_obs'], 
#                       expert_subsets=env.expert_subsets, update_trick = True)
#     PMML_EWA_agent = algos.PMML_EWA(n_bandits=N_BANDITS, horizon=HORIZON, n_tasks=N_TASKS,
#                       expert_subsets=env.expert_subsets)
    return {
            'moss_agent':moss_agent,
            'EE_agent':EE_agent,
            'opt_moss_agent':opt_moss_agent,
            'PMML_agent':PMML_agent,
#             'exp3_agent':exp3_agent,
#             'exp3_reset_agent':exp3_reset_agent,
#             'EWAmaxStats_agent':EWAmaxStats_agent,
#             'EWAmaxStatsTrick_agent':EWAmaxStatsTrick_agent,
#             'PMML_EWA_agent':PMML_EWA_agent,
           }

def _init_cache(N_EXPS, x_axis):
    moss_regrets = np.zeros((N_EXPS, x_axis))
    EE_regrets = np.zeros((N_EXPS, x_axis))
    PMML_regrets = np.zeros((N_EXPS, x_axis))
    opt_moss_regrets = np.zeros((N_EXPS, x_axis))
#     exp3_reset_regrets = np.zeros((N_EXPS, x_axis))
#     EWAmaxStats_regrets = np.zeros((N_EXPS, x_axis))
#     EWAmaxStatsTrick_regrets = np.zeros((N_EXPS, x_axis))
#     exp3_regrets = np.zeros((N_EXPS, x_axis))
#     PMML_EWA_regrets = np.zeros((N_EXPS, x_axis))
    return {
        'moss_regrets':moss_regrets,
        'EE_regrets':EE_regrets,
        'PMML_regrets':PMML_regrets,
        'opt_moss_regrets':opt_moss_regrets,
#         'exp3_reset_regrets':exp3_reset_regrets,
#         'EWAmaxStats_regrets':EWAmaxStats_regrets,
#         'EWAmaxStatsTrick_regrets':EWAmaxStatsTrick_regrets,
#         'exp3_regrets':exp3_regrets,
#         'PMML_EWA_regrets':PMML_EWA_regrets,
    }

def _collect_data(agent_dict, cache_dict, i, j, n_tasks, HORIZON, quiet, env, exp_type):
    moss_r = meta_rolls_out(n_tasks, agent_dict['moss_agent'], env, HORIZON, quiet)
    EE_r= meta_rolls_out(n_tasks, agent_dict['EE_agent'], env, HORIZON, quiet)
    PMML_r= meta_rolls_out(n_tasks, agent_dict['PMML_agent'], env, HORIZON, quiet)
    opt_moss_r = meta_rolls_out(n_tasks, agent_dict['opt_moss_agent'], env, HORIZON, quiet)
#     EWAmaxStats_r = meta_rolls_out(n_tasks, agent_dict['EWAmaxStats_agent'], env, HORIZON, quiet)
#     PMML_r = meta_rolls_out(n_tasks, agent_dict['EWAmaxStatsTrick_agent'], env, HORIZON, quiet)
#     exp3_r = meta_rolls_out(n_tasks, agent_dict['exp3_agent'], env, HORIZON, quiet)
#     exp3_reset_r = meta_rolls_out(n_tasks, agent_dict['exp3_reset_agent'], env, HORIZON, quiet)
#     PMML_EWA_r = meta_rolls_out(n_tasks, agent_dict['PMML_EWA_agent'], env, HORIZON, quiet)
    if exp_type == TASK_EXP:
        cache_dict['moss_regrets'][i] = moss_r
        cache_dict['EE_regrets'][i] = EE_r
        cache_dict['PMML_regrets'][i] = PMML_r
        cache_dict['opt_moss_regrets'][i] = opt_moss_r
#         cache_dict['EWAmaxStats_regrets'][i] = EWAmaxStats_r
#         cache_dict['EWAmaxStatsTrick_regrets'][i] = PMML_r
#         cache_dict['exp3_regrets'][i] = exp3_r
#         cache_dict['exp3_reset_regrets'][i] = exp3_reset_r
#         cache_dict['PMML_EWA_regrets'][i] = PMML_EWA_r
    elif exp_type == HORIZON_EXP:
        cache_dict['moss_regrets'][i, j] = moss_r[-1]/HORIZON
        cache_dict['EE_regrets'][i, j] = EE_r[-1]/HORIZON
        cache_dict['PMML_regrets'][i, j] = PMML_r[-1]/HORIZON
        cache_dict['opt_moss_regrets'][i, j] = opt_moss_r[-1]/HORIZON
#         cache_dict['EWAmaxStats_regrets'][i, j] = EWAmaxStats_r[-1]/HORIZON
#         cache_dict['EWAmaxStatsTrick_regrets'][i, j] = PMML_r[-1]/HORIZON
#         cache_dict['exp3_regrets'][i, j] = exp3_r[-1]/HORIZON
#         cache_dict['exp3_reset_regrets'][i, j] = exp3_reset_r[-1]/HORIZON
#         cache_dict['PMML_EWA_regrets'][i, j] = PMML_EWA_r[-1]/HORIZON
    else:
        cache_dict['moss_regrets'][i, j] = moss_r[-1]
        cache_dict['EE_regrets'][i, j] = EE_r[-1]
        cache_dict['PMML_regrets'][i, j] = PMML_r[-1]
        cache_dict['opt_moss_regrets'][i, j] = opt_moss_r[-1]
#         cache_dict['EWAmaxStats_regrets'][i, j] = EWAmaxStats_r[-1]
#         cache_dict['EWAmaxStatsTrick_regrets'][i, j] = PMML_r[-1]
#         cache_dict['exp3_regrets'][i, j] = exp3_r[-1]
#         cache_dict['exp3_reset_regrets'][i, j] = exp3_reset_r[-1]
#         cache_dict['PMML_EWA_regrets'][i, j] = PMML_EWA_r[-1]
    return cache_dict

def task_exp(N_EXPS, N_TASKS, N_BANDITS, HORIZON, OPT_SIZE, N_EXPERT, DS_NAME, quiet=True, **kwargs):
    env = bandit.MetaBernoulli(n_bandits=N_BANDITS, opt_size=OPT_SIZE, n_tasks=N_TASKS, 
                           n_experts=N_EXPERT, ds_name=DS_NAME, **kwargs)
    cache_dict = _init_cache(N_EXPS, N_TASKS)
    for i in trange(N_EXPS):
        agent_dict = _init_agents(N_EXPS, N_TASKS, N_BANDITS, HORIZON, OPT_SIZE, N_EXPERT, DS_NAME, env, quiet, **kwargs)
        cache_dict = _collect_data(agent_dict, cache_dict, i, None, N_TASKS, HORIZON, quiet, env, TASK_EXP)
    X = np.arange(N_TASKS)
    if N_EXPERT is None:
        N_EXPERT = env.n_experts
    gap = kwargs['gap_constrain']
    title = f'Regret: {N_BANDITS} arms, horizon {HORIZON}, {N_EXPERT} experts, gap = {gap:.3f} and subset size {OPT_SIZE}'
    xlabel, ylabel = 'Number of tasks', 'Average Regret per task'
    step = kwargs['task_cache_step']
    indices = np.arange(0, X.shape[0], step).astype(int)
    regret_dict = {
        'moss_regrets':cache_dict['moss_regrets'][:,indices],
        'EE_regrets':cache_dict['EE_regrets'][:,indices],
        'PMML_regrets':cache_dict['PMML_regrets'][:,indices],
        'opt_moss_regrets':cache_dict['opt_moss_regrets'][:,indices],
#         'EWAmaxStats_regrets':cache_dict['EWAmaxStats_regrets'][:,indices],
#         'EWAmaxStatsTrick_regrets':cache_dict['EWAmaxStatsTrick_regrets'][:,indices],
#         'exp3_regrets':cache_dict['exp3_regrets'][:,indices],
#         'exp3_reset_regrets':cache_dict['exp3_reset_regrets'][:,indices],
#         'PMML_EWA_regrets':cache_dict['PMML_EWA_regrets'][:,indices],
    }
    plot(X[indices], regret_dict, title, xlabel, ylabel, kwargs['plot_var'])
    return (X, regret_dict, title, xlabel, ylabel)


def horizon_exp(N_EXPS, N_TASKS, N_BANDITS, OPT_SIZE, N_EXPERT, DS_NAME, horizon_list = np.arange(1,202,50)*10, quiet=True, **kwargs):
    cache_dict = _init_cache(N_EXPS, horizon_list.shape[0])
    for i in trange(N_EXPS):
        for j, h in enumerate(horizon_list):
            kwargs['gap_constrain'] = min(1,np.sqrt(N_BANDITS*np.log(N_TASKS)/h))
            tmp=kwargs['gap_constrain']
            print(f'gap = {tmp}')
            env = bandit.MetaBernoulli(n_bandits=N_BANDITS, opt_size=OPT_SIZE, n_tasks=N_TASKS, 
                                   n_experts=N_EXPERT, ds_name=DS_NAME, **kwargs)
            agent_dict = _init_agents(N_EXPS, N_TASKS, N_BANDITS, h, OPT_SIZE, N_EXPERT, DS_NAME, env, quiet, **kwargs)
            cache_dict = _collect_data(agent_dict, cache_dict, i, j, N_TASKS, h, quiet, env, HORIZON_EXP)
    X = horizon_list
    if N_EXPERT is None:
        N_EXPERT = env.n_experts
    title = f'Regret: {N_BANDITS} arms, {N_TASKS} tasks, {N_EXPERT} experts, gap cond. satisfied and subset size {OPT_SIZE}'
    xlabel, ylabel = 'Horizon', 'Average Regret per Step'
    regret_dict = {
        'moss_regrets':cache_dict['moss_regrets'],
        'EE_regrets':cache_dict['EE_regrets'],
        'PMML_regrets':cache_dict['PMML_regrets'],
        'opt_moss_regrets':cache_dict['opt_moss_regrets'],
#         'EWAmaxStats_regrets':cache_dict['EWAmaxStats_regrets'],
#         'EWAmaxStatsTrick_regrets':cache_dict['EWAmaxStatsTrick_regrets'],
#         'exp3_regrets':cache_dict['exp3_regrets'],
#         'exp3_reset_regrets':cache_dict['exp3_reset_regrets'],
#         'PMML_EWA_regrets':cache_dict['PMML_EWA_regrets'],
    }
    plot(X, regret_dict, title, xlabel, ylabel, kwargs['plot_var'])
    return (X, regret_dict, title, xlabel, ylabel)


def arm_exp(N_EXPS, N_TASKS, HORIZON, OPT_SIZE, N_EXPERT, DS_NAME, n_bandits_list = np.arange(8,69,15), quiet=True, **kwargs):
    cache_dict = _init_cache(N_EXPS, n_bandits_list.shape[0])
    for i in trange(N_EXPS):
        for j, b in enumerate(n_bandits_list):
            kwargs['gap_constrain'] = min(1,np.sqrt(b*np.log(N_TASKS)/HORIZON))
            env = bandit.MetaBernoulli(n_bandits=b, opt_size=OPT_SIZE, n_tasks=N_TASKS, 
                               n_experts=N_EXPERT, ds_name=DS_NAME, **kwargs)
            agent_dict = _init_agents(N_EXPS, N_TASKS, b, HORIZON, OPT_SIZE, N_EXPERT, DS_NAME, env, quiet, **kwargs)
            cache_dict = _collect_data(agent_dict, cache_dict, i, j, N_TASKS, HORIZON, quiet, env, ARM_EXP)
    X = n_bandits_list
    if N_EXPERT is None:
        N_EXPERT = 'all'
    title = f'Regret: Horizon {HORIZON}, {N_TASKS} tasks, {N_EXPERT} experts, gap cond. satisfied and subset size {OPT_SIZE}'
    xlabel, ylabel = 'Number of Arms', 'Regret'
    regret_dict = {
        'moss_regrets':cache_dict['moss_regrets'],
        'EE_regrets':cache_dict['EE_regrets'],
        'PMML_regrets':cache_dict['PMML_regrets'],
        'opt_moss_regrets':cache_dict['opt_moss_regrets'],
#         'EWAmaxStats_regrets':cache_dict['EWAmaxStats_regrets'],
#         'EWAmaxStatsTrick_regrets':cache_dict['EWAmaxStatsTrick_regrets'],
#         'exp3_regrets':cache_dict['exp3_regrets'],
#         'exp3_reset_regrets':cache_dict['exp3_reset_regrets'],
#         'PMML_EWA_regrets':cache_dict['PMML_EWA_regrets'],
    }
    plot(X, regret_dict, title, xlabel, ylabel, kwargs['plot_var'])
    return (X, regret_dict, title, xlabel, ylabel)


def subset_size_exp(N_EXPS, N_TASKS, N_BANDITS, HORIZON, N_EXPERT, DS_NAME, opt_size_list = None, quiet=True, **kwargs):
    if opt_size_list is None:
        opt_size_list = np.arange(1,N_BANDITS+1,4)
    cache_dict = _init_cache(N_EXPS, opt_size_list.shape[0])
    for i in trange(N_EXPS):
        for j, s in enumerate(opt_size_list):
            env = bandit.MetaBernoulli(n_bandits=N_BANDITS, opt_size=s, n_tasks=N_TASKS, 
                                       n_experts=N_EXPERT, ds_name=DS_NAME, **kwargs)
            agent_dict = _init_agents(N_EXPS, N_TASKS, N_BANDITS, HORIZON, s, N_EXPERT, DS_NAME, env, quiet, **kwargs)
            cache_dict = _collect_data(agent_dict, cache_dict, i, j, N_TASKS, HORIZON, quiet, env, SUBSET_EXP)
    X = opt_size_list
    if N_EXPERT is None:
        N_EXPERT = 'all'
    gap = kwargs['gap_constrain']
    title=f'Regret: {N_BANDITS} arms, Horizon {HORIZON}, {N_TASKS} tasks, gap = {gap:.3f} and {N_EXPERT} experts'
    xlabel, ylabel = 'subset size', 'Regret'
    regret_dict = {
        'moss_regrets':cache_dict['moss_regrets'],
        'EE_regrets':cache_dict['EE_regrets'],
        'PMML_regrets':cache_dict['PMML_regrets'],
        'opt_moss_regrets':cache_dict['opt_moss_regrets'],
#         'EWAmaxStats_regrets':cache_dict['EWAmaxStats_regrets'],
#         'EWAmaxStatsTrick_regrets':cache_dict['EWAmaxStatsTrick_regrets'],
#         'exp3_regrets':cache_dict['exp3_regrets'],
#         'exp3_reset_regrets':cache_dict['exp3_reset_regrets'],
#         'PMML_EWA_regrets':cache_dict['PMML_EWA_regrets'],
    }
    plot(X, regret_dict, title, xlabel, ylabel, kwargs['plot_var'])
    return (X, regret_dict, title, xlabel, ylabel)
