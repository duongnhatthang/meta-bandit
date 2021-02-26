import numpy as np
import matplotlib.pyplot as plt
import algos
import bandit
from tqdm import tqdm, trange

def rolls_out(agent, env, horizon, quiet):
#     observations = []
#     actions = []
    rewards = []
#     next_observations = []
    obs = env.reset()        
    if quiet == False:
        for i in trange(horizon):
            a = agent.get_action(obs)
            next_obs, r, _, _ = env.step(a)
    #         observations.append(obs)
            rewards.append(r)
    #         actions.append(a)
    #         next_observations.append(next_obs)
            obs = next_obs
            if hasattr(agent, 'update'): #For EXP3
                agent.update(a, r)
    else:
        for i in range(horizon):
            a = agent.get_action(obs)
            next_obs, r, _, _ = env.step(a)
    #         observations.append(obs)
            rewards.append(r)
    #         actions.append(a)
    #         next_observations.append(next_obs)
            obs = next_obs
            if hasattr(agent, 'update'): #For EXP3
                agent.update(a, r)
    if hasattr(agent, 'eps_end_update'): #For metaAlg
        agent.eps_end_update(obs)
    if isinstance(agent, algos.Exp3) == True and agent.is_reset == True:
        agent.reset()
    regret = np.max(env._p)*horizon - np.sum(rewards)
#     return regret, dict(
#         observations=np.array(observations),
#         actions=np.array(actions),
#         rewards=np.array(rewards),
#         next_observations=np.array(next_observations),
#     )
    return regret, None

def meta_rolls_out(n_switches, agent, env, horizon, quiet):
    regrets = []
    tmp_regrets = []
    caches = []
    for idx in range(n_switches+1):
        env.reset_task(idx)
#         print(f'Env: {env._p}')
        r, c = rolls_out(agent, env, horizon, quiet)
        tmp_regrets.append(r)
#         caches.append(c)
        regrets.append(np.average(tmp_regrets)) #average regret until this switch
    return regrets, caches

def plot(X, meta_regrets, exp3_regrets, moss_regrets, exp3_reset_regrets, meta_trick_regrets, metaPE_regrets, title, xlabel, ylabel):
    meta_Y = np.mean(meta_regrets, axis=0)
    meta_dY = 2*np.sqrt(np.var(meta_regrets, axis=0))
    metaPE_Y = np.mean(metaPE_regrets, axis=0)
    metaPE_dY = 2*np.sqrt(np.var(metaPE_regrets, axis=0))
    meta_trick_Y = np.mean(meta_trick_regrets, axis=0)
    meta_trick_dY = 2*np.sqrt(np.var(meta_trick_regrets, axis=0))
    exp3_Y = np.mean(exp3_regrets, axis=0)
    exp3_dY = 2*np.sqrt(np.var(exp3_regrets, axis=0))
    exp3_reset_Y = np.mean(exp3_reset_regrets, axis=0)
    exp3_reset_dY = 2*np.sqrt(np.var(exp3_reset_regrets, axis=0))
    moss_Y = np.mean(moss_regrets, axis=0)
    moss_dY = 2*np.sqrt(np.var(moss_regrets, axis=0))
    plt.plot(X, metaPE_Y, '-', color='orange', label = "Phase Elimination")
    plt.fill_between(X, metaPE_Y - metaPE_dY, metaPE_Y + metaPE_dY, color='orange', alpha=0.2)
    plt.plot(X, meta_Y, '-', color='red', label = "EWA")
    plt.fill_between(X, meta_Y - meta_dY, meta_Y + meta_dY, color='red', alpha=0.2)
    plt.plot(X, meta_trick_Y, '-', color='black', label = "EWA (trick)")
    plt.fill_between(X, meta_trick_Y - meta_trick_dY, meta_trick_Y + meta_trick_dY, color='black', alpha=0.2)
    plt.plot(X, exp3_Y, '-', color='blue', label = "EXP3")
    plt.fill_between(X, exp3_Y - exp3_dY, exp3_Y + exp3_dY, color='blue', alpha=0.2)
    plt.plot(X, moss_Y, '-', color='green', label = "MOSS")
    plt.fill_between(X, moss_Y - moss_dY, moss_Y + moss_dY, color='green', alpha=0.2)
    plt.plot(X, exp3_reset_Y, '-', color='yellow', label = "EXP3 Reset")
    plt.fill_between(X, exp3_reset_Y - exp3_reset_dY, exp3_reset_Y + exp3_reset_dY, color='yellow', alpha=0.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.yscale('log')
    plt.legend()
    plt.title(title)


def switches_exp(N_EXPS, N_SWITCHES, N_BANDITS, HORIZON, N_UNBIASED_OBS, ALG_NAME, OPT_SIZE, N_EXPERT, DS_NAME, quiet=True, **kwargs):
    env = bandit.MetaBernoulli(n_bandits=N_BANDITS, opt_size=OPT_SIZE, n_tasks=N_SWITCHES+1, 
                           n_experts=N_EXPERT, ds_name=DS_NAME, **kwargs)
    exp3_kwargs = {'n_switches':N_SWITCHES}
    exp3_agent = algos.Exp3(n_bandits=N_BANDITS, horizon=HORIZON, **exp3_kwargs)
    moss_agent = algos.MOSS(n_bandits=N_BANDITS, horizon=HORIZON)
    exp3_reset_agent = algos.Exp3(n_bandits=N_BANDITS, horizon=HORIZON, is_reset=True)
    exp3_caches, exp3_reset_caches, moss_caches, meta_caches, metaPE_caches, meta_trick_caches = [], [], [], [], [], []
    exp3_reset_regrets = np.zeros((N_EXPS, N_SWITCHES+1))
    meta_regrets = np.zeros((N_EXPS, N_SWITCHES+1))
    metaPE_regrets = np.zeros((N_EXPS, N_SWITCHES+1))
    meta_trick_regrets = np.zeros((N_EXPS, N_SWITCHES+1))
    exp3_regrets = np.zeros((N_EXPS, N_SWITCHES+1))
    moss_regrets = np.zeros((N_EXPS, N_SWITCHES+1))
    for i in trange(N_EXPS):
        meta_agent = algos.MetaAlg(n_bandits=N_BANDITS, horizon=HORIZON, n_switches=N_SWITCHES, 
                          n_unbiased_obs=N_UNBIASED_OBS, alg_name=ALG_NAME, 
                          expert_subgroups=env.expert_subgroups)
        meta_trick_agent = algos.MetaAlg(n_bandits=N_BANDITS, horizon=HORIZON, n_switches=N_SWITCHES, 
                          n_unbiased_obs=N_UNBIASED_OBS, alg_name=ALG_NAME, 
                          expert_subgroups=env.expert_subgroups, update_trick = True)
        metaPE_agent = algos.MetaPE(n_bandits=N_BANDITS, horizon=HORIZON, n_switches=N_SWITCHES,
                          expert_subgroups=env.expert_subgroups)
        meta_r, meta_c = meta_rolls_out(N_SWITCHES, meta_agent, env, HORIZON, quiet)
        meta_caches.append(meta_c)
        meta_regrets[i] = meta_r
        meta_trick_r, meta_trick_c = meta_rolls_out(N_SWITCHES, meta_trick_agent, env, HORIZON, quiet)
        meta_trick_caches.append(meta_trick_c)
        meta_trick_regrets[i] = meta_trick_r
        metaPE_r, metaPE_c = meta_rolls_out(N_SWITCHES, metaPE_agent, env, HORIZON, quiet)
        metaPE_caches.append(metaPE_c)
        metaPE_regrets[i] = metaPE_r
        exp3_r, exp3_c = meta_rolls_out(N_SWITCHES, exp3_agent, env, HORIZON, quiet)
        exp3_caches.append(exp3_c)
        exp3_regrets[i] = exp3_r
        moss_r, moss_c = meta_rolls_out(N_SWITCHES, moss_agent, env, HORIZON, quiet)
        moss_caches.append(moss_c)
        moss_regrets[i] = moss_r
        exp3_reset_r, exp3_reset_c = meta_rolls_out(N_SWITCHES, exp3_reset_agent, env, HORIZON, quiet)
        exp3_reset_caches.append(exp3_reset_c)
        exp3_reset_regrets[i] = exp3_reset_r

    X = np.arange(N_SWITCHES+1)
    if N_EXPERT is None:
        N_EXPERT = env.n_experts
    gap = kwargs['gap_constrain']
    title = f'Regret: {N_BANDITS} arms, horizon {HORIZON}, {N_EXPERT} experts, gap = {gap} and subgroup size {OPT_SIZE}'
    xlabel, ylabel = 'Number of switches', 'Average Regret per Episode'
    step = kwargs['switches_cache_step']
    indices = np.arange(0, X.shape[0], step).astype(int)
    plot(X[indices], meta_regrets[:,indices], exp3_regrets[:,indices], moss_regrets[:,indices], exp3_reset_regrets[:,indices], meta_trick_regrets[:,indices], metaPE_regrets[:,indices], title, xlabel, ylabel)
    return (X, meta_regrets, exp3_regrets, moss_regrets, exp3_reset_regrets, meta_trick_regrets, metaPE_regrets, title, xlabel, ylabel)


def horizon_exp(N_EXPS, N_SWITCHES, N_BANDITS, N_UNBIASED_OBS, ALG_NAME, OPT_SIZE, N_EXPERT, DS_NAME, horizon_list = np.arange(1,202,50)*10, quiet=True, **kwargs):
    env = bandit.MetaBernoulli(n_bandits=N_BANDITS, opt_size=OPT_SIZE, n_tasks=N_SWITCHES+1, 
                           n_experts=N_EXPERT, ds_name=DS_NAME, **kwargs)
    exp3_caches, exp3_reset_caches, moss_caches, meta_caches, metaPE_caches, meta_trick_caches = [], [], [], [], [], []
    exp3_reset_regrets = np.zeros((N_EXPS, horizon_list.shape[0]))
    meta_regrets = np.zeros((N_EXPS, horizon_list.shape[0]))
    metaPE_regrets = np.zeros((N_EXPS, horizon_list.shape[0]))
    meta_trick_regrets = np.zeros((N_EXPS, horizon_list.shape[0]))
    exp3_regrets = np.zeros((N_EXPS, horizon_list.shape[0]))
    moss_regrets = np.zeros((N_EXPS, horizon_list.shape[0]))
    exp3_kwargs = {'n_switches':N_SWITCHES}
    for i in trange(N_EXPS):
        for j, h in enumerate(horizon_list):
            exp3_agent = algos.Exp3(n_bandits=N_BANDITS, horizon=h, **exp3_kwargs)
            exp3_reset_agent = algos.Exp3(n_bandits=N_BANDITS, horizon=h, is_reset=True)
            moss_agent = algos.MOSS(n_bandits=N_BANDITS, horizon=h)
            meta_agent = algos.MetaAlg(n_bandits=N_BANDITS, horizon=h, n_switches=N_SWITCHES, 
                              n_unbiased_obs=N_UNBIASED_OBS, alg_name=ALG_NAME, 
                              expert_subgroups=env.expert_subgroups)
            meta_trick_agent = algos.MetaAlg(n_bandits=N_BANDITS, horizon=h, n_switches=N_SWITCHES, 
                              n_unbiased_obs=N_UNBIASED_OBS, alg_name=ALG_NAME, 
                              expert_subgroups=env.expert_subgroups, update_trick = True)
            metaPE_agent = algos.MetaPE(n_bandits=N_BANDITS, horizon=h, n_switches=N_SWITCHES,
                              expert_subgroups=env.expert_subgroups)
            meta_r, meta_c = meta_rolls_out(N_SWITCHES, meta_agent, env, h, quiet)
            meta_caches.append(meta_c)
            meta_regrets[i, j] = meta_r[-1]/h
            metaPE_r, metaPE_c = meta_rolls_out(N_SWITCHES, metaPE_agent, env, h, quiet)
            metaPE_caches.append(metaPE_c)
            metaPE_regrets[i, j] = metaPE_r[-1]/h
            meta_trick_r, meta_trick_c = meta_rolls_out(N_SWITCHES, meta_trick_agent, env, h, quiet)
            meta_trick_caches.append(meta_trick_c)
            meta_trick_regrets[i, j] = meta_trick_r[-1]/h
            exp3_r, exp3_c = meta_rolls_out(N_SWITCHES, exp3_agent, env, h, quiet)
            exp3_caches.append(exp3_c)
            exp3_regrets[i, j] = exp3_r[-1]/h
            moss_r, moss_c = meta_rolls_out(N_SWITCHES, moss_agent, env, h, quiet)
            moss_caches.append(moss_c)
            moss_regrets[i, j] = moss_r[-1]/h
            exp3_reset_r, exp3_reset_c = meta_rolls_out(N_SWITCHES, exp3_reset_agent, env, h, quiet)
            exp3_reset_caches.append(exp3_reset_c)
            exp3_reset_regrets[i, j] = exp3_reset_r[-1]/h

    X = horizon_list
    if N_EXPERT is None:
        N_EXPERT = env.n_experts
    gap = kwargs['gap_constrain']
    title = f'Regret: {N_BANDITS} arms, {N_SWITCHES} switches, {N_EXPERT} experts, gap = {gap} and subgroup size {OPT_SIZE}'
    xlabel, ylabel = 'Horizon', 'Average Regret per Step'
    plot(X, meta_regrets, exp3_regrets, moss_regrets, exp3_reset_regrets, meta_trick_regrets, metaPE_regrets, title, xlabel, ylabel)
    return (X, meta_regrets, exp3_regrets, moss_regrets, exp3_reset_regrets, meta_trick_regrets, metaPE_regrets, title, xlabel, ylabel)


def arm_exp(N_EXPS, N_SWITCHES, HORIZON, N_UNBIASED_OBS, ALG_NAME, OPT_SIZE, N_EXPERT, DS_NAME, n_bandits_list = np.arange(8,69,15), quiet=True, **kwargs):
    exp3_caches, exp3_reset_caches, moss_caches, meta_caches, metaPE_caches, meta_trick_caches = [], [], [], [], [], []
    meta_regrets = np.zeros((N_EXPS, n_bandits_list.shape[0]))
    metaPE_regrets = np.zeros((N_EXPS, n_bandits_list.shape[0]))
    meta_trick_regrets= np.zeros((N_EXPS, n_bandits_list.shape[0]))
    exp3_regrets= np.zeros((N_EXPS, n_bandits_list.shape[0]))
    exp3_reset_regrets= np.zeros((N_EXPS, n_bandits_list.shape[0]))
    moss_regrets= np.zeros((N_EXPS, n_bandits_list.shape[0]))
    exp3_kwargs = {'n_switches':N_SWITCHES}
    for i in trange(N_EXPS):
        for j, b in enumerate(tqdm(n_bandits_list)):
            env = bandit.MetaBernoulli(n_bandits=b, opt_size=OPT_SIZE, n_tasks=N_SWITCHES+1, 
                               n_experts=N_EXPERT, ds_name=DS_NAME, **kwargs)
            exp3_agent = algos.Exp3(n_bandits=b, horizon=HORIZON, **exp3_kwargs)
            exp3_reset_agent = algos.Exp3(n_bandits=b, horizon=HORIZON, is_reset=True)
            moss_agent = algos.MOSS(n_bandits=b, horizon=HORIZON)
            meta_agent = algos.MetaAlg(n_bandits=b, horizon=HORIZON, n_switches=N_SWITCHES, 
                              n_unbiased_obs=N_UNBIASED_OBS, alg_name=ALG_NAME, 
                              expert_subgroups=env.expert_subgroups)
            meta_trick_agent = algos.MetaAlg(n_bandits=b, horizon=HORIZON, n_switches=N_SWITCHES, 
                              n_unbiased_obs=N_UNBIASED_OBS, alg_name=ALG_NAME, 
                              expert_subgroups=env.expert_subgroups, update_trick = True)
            metaPE_agent = algos.MetaPE(n_bandits=b, horizon=HORIZON, n_switches=N_SWITCHES,
                              expert_subgroups=env.expert_subgroups)
            moss_r, moss_c = meta_rolls_out(N_SWITCHES, moss_agent, env, HORIZON, quiet)
            moss_caches.append(moss_c)
            moss_regrets[i, j] = moss_r[-1]
            meta_r, meta_c = meta_rolls_out(N_SWITCHES, meta_agent, env, HORIZON, quiet)
            meta_caches.append(meta_c)
            meta_regrets[i, j] = meta_r[-1]
            metaPE_r, metaPE_c = meta_rolls_out(N_SWITCHES, metaPE_agent, env, HORIZON, quiet)
            metaPE_caches.append(metaPE_c)
            metaPE_regrets[i, j] = metaPE_r[-1]
            meta_trick_r, meta_trick_c = meta_rolls_out(N_SWITCHES, meta_trick_agent, env, HORIZON, quiet)
            meta_trick_caches.append(meta_trick_c)
            meta_trick_regrets[i, j] = meta_trick_r[-1]
            exp3_r, exp3_c = meta_rolls_out(N_SWITCHES, exp3_agent, env, HORIZON, quiet)
            exp3_caches.append(exp3_c)
            exp3_regrets[i, j] = exp3_r[-1]
            exp3_reset_r, exp3_reset_c = meta_rolls_out(N_SWITCHES, exp3_reset_agent, env, HORIZON, quiet)
            exp3_reset_caches.append(exp3_reset_c)
            exp3_reset_regrets[i, j] = exp3_reset_r[-1]
    X = n_bandits_list
    if N_EXPERT is None:
        N_EXPERT = 'All'
    gap = kwargs['gap_constrain']
    title = f'Regret: Horizon {HORIZON}, {N_SWITCHES} switches, {N_EXPERT} experts, gap = {gap} and subgroup size {OPT_SIZE}'
    xlabel, ylabel = 'Number of Arms', 'Regret'
    plot(X, meta_regrets, exp3_regrets, moss_regrets, exp3_reset_regrets, meta_trick_regrets, metaPE_regrets, title, xlabel, ylabel)
    return (X, meta_regrets, exp3_regrets, moss_regrets, exp3_reset_regrets, meta_trick_regrets, metaPE_regrets, title, xlabel, ylabel)


def subgroup_size_exp(N_EXPS, N_SWITCHES, N_BANDITS, HORIZON, N_UNBIASED_OBS, ALG_NAME, N_EXPERT, DS_NAME, opt_size_list = None, quiet=True, **kwargs):
    if opt_size_list is None:
        opt_size_list = np.arange(1,N_BANDITS+1,4)
    exp3_caches, exp3_reset_caches, moss_caches, meta_caches, metaPE_caches, meta_trick_caches= [], [], [], [], [], []
    meta_regrets = np.zeros((N_EXPS, opt_size_list.shape[0]))
    metaPE_regrets = np.zeros((N_EXPS, opt_size_list.shape[0]))
    meta_trick_regrets = np.zeros((N_EXPS, opt_size_list.shape[0]))
    exp3_regrets = np.zeros((N_EXPS, opt_size_list.shape[0]))
    exp3_reset_regrets = np.zeros((N_EXPS, opt_size_list.shape[0]))
    moss_regrets = np.zeros((N_EXPS, opt_size_list.shape[0]))
    exp3_kwargs = {'n_switches':N_SWITCHES}
    for i in trange(N_EXPS):
        for j, s in enumerate(tqdm(opt_size_list)):
            env = bandit.MetaBernoulli(n_bandits=N_BANDITS, opt_size=s, n_tasks=N_SWITCHES+1, 
                                       n_experts=N_EXPERT, ds_name=DS_NAME, **kwargs)
            exp3_agent = algos.Exp3(n_bandits=N_BANDITS, horizon=HORIZON, **exp3_kwargs)
            exp3_reset_agent = algos.Exp3(n_bandits=N_BANDITS, horizon=HORIZON, is_reset=True)
            moss_agent = algos.MOSS(n_bandits=N_BANDITS, horizon=HORIZON)
            meta_agent = algos.MetaAlg(n_bandits=N_BANDITS, horizon=HORIZON, n_switches=N_SWITCHES, 
                              n_unbiased_obs=N_UNBIASED_OBS, alg_name=ALG_NAME, 
                              expert_subgroups=env.expert_subgroups)
            meta_trick_agent = algos.MetaAlg(n_bandits=N_BANDITS, horizon=HORIZON, n_switches=N_SWITCHES, 
                              n_unbiased_obs=N_UNBIASED_OBS, alg_name=ALG_NAME, 
                              expert_subgroups=env.expert_subgroups, update_trick = True)
            metaPE_agent = algos.MetaPE(n_bandits=N_BANDITS, horizon=HORIZON, n_switches=N_SWITCHES,
                              expert_subgroups=env.expert_subgroups)
            meta_r, meta_c = meta_rolls_out(N_SWITCHES, meta_agent, env, HORIZON, quiet)
            meta_caches.append(meta_c)
            meta_regrets[i, j] = meta_r[-1]
            metaPE_r, metaPE_c = meta_rolls_out(N_SWITCHES, metaPE_agent, env, HORIZON, quiet)
            metaPE_caches.append(metaPE_c)
            metaPE_regrets[i, j] = metaPE_r[-1]
            meta_trick_r, meta_trick_c = meta_rolls_out(N_SWITCHES, meta_trick_agent, env, HORIZON, quiet)
            meta_trick_caches.append(meta_trick_c)
            meta_trick_regrets[i, j] = meta_trick_r[-1]
            exp3_r, exp3_c = meta_rolls_out(N_SWITCHES, exp3_agent, env, HORIZON, quiet)
            exp3_caches.append(exp3_c)
            exp3_regrets[i, j] = exp3_r[-1]
            moss_r, moss_c = meta_rolls_out(N_SWITCHES, moss_agent, env, HORIZON, quiet)
            moss_caches.append(moss_c)
            moss_regrets[i, j] = moss_r[-1]
            exp3_reset_r, exp3_reset_c = meta_rolls_out(N_SWITCHES, exp3_reset_agent, env, HORIZON, quiet)
            exp3_reset_caches.append(exp3_reset_c)
            exp3_reset_regrets[i, j] = exp3_reset_r[-1]

    X = opt_size_list
    if N_EXPERT is None:
        N_EXPERT = env.n_experts
    gap = kwargs['gap_constrain']
    title=f'Regret: {N_BANDITS} arms, Horizon {HORIZON}, {N_SWITCHES} switches, gap = {gap} and {N_EXPERT} experts'
    xlabel, ylabel = 'Subgroup size', 'Regret'
    plot(X, meta_regrets, exp3_regrets, moss_regrets, exp3_reset_regrets, meta_trick_regrets, metaPE_regrets, title, xlabel, ylabel)
    return (X, meta_regrets, exp3_regrets, moss_regrets, exp3_reset_regrets, meta_trick_regrets, metaPE_regrets, title, xlabel, ylabel)


# def experts_exp(N_EXPS, N_SWITCHES, N_BANDITS, HORIZON, N_UNBIASED_OBS, ALG_NAME, OPT_SIZE, DS_NAME, n_experts_list = None, quiet=True, **kwargs):
#     if n_experts_list is None:
#         max_n_expert = len(list(combinations(np.arange(N_BANDITS),OPT_SIZE)))
#         n_experts_list = np.arange(10,min(411,max_n_expert),20)
#     exp3_caches, exp3_reset_caches, moss_caches, meta_caches, meta_trick_caches = [], [], [], [], []
#     meta_regrets = np.zeros((N_EXPS, n_experts_list.shape[0]))
#     meta_trick_regrets = np.zeros((N_EXPS, n_experts_list.shape[0]))
#     exp3_regrets = np.zeros((N_EXPS, n_experts_list.shape[0]))
#     exp3_reset_regrets = np.zeros((N_EXPS, n_experts_list.shape[0]))
#     moss_regrets = np.zeros((N_EXPS, n_experts_list.shape[0]))
#     exp3_kwargs = {'n_switches':N_SWITCHES}
#     for i in trange(N_EXPS):
#         for j, e in enumerate(tqdm(n_experts_list)):
#             env = bandit.MetaBernoulli(n_bandits=N_BANDITS, opt_size=OPT_SIZE, n_tasks=N_SWITCHES+1, 
#                                        n_experts=e, ds_name=DS_NAME, **kwargs)
#             exp3_agent = algos.Exp3(n_bandits=N_BANDITS, horizon=HORIZON, **exp3_kwargs)
#             exp3_reset_agent = algos.Exp3(n_bandits=N_BANDITS, horizon=HORIZON, is_reset=True)
#             moss_agent = algos.MOSS(n_bandits=N_BANDITS, horizon=HORIZON)
#             meta_agent = algos.MetaAlg(n_bandits=N_BANDITS, horizon=HORIZON, n_switches=N_SWITCHES, 
#                               n_unbiased_obs=N_UNBIASED_OBS, alg_name=ALG_NAME, 
#                               expert_subgroups=env.expert_subgroups)
#             meta_trick_agent = algos.MetaAlg(n_bandits=N_BANDITS, horizon=HORIZON, n_switches=N_SWITCHES, 
#                               n_unbiased_obs=N_UNBIASED_OBS, alg_name=ALG_NAME, 
#                               expert_subgroups=env.expert_subgroups, update_trick = True)
#             meta_r, meta_c = meta_rolls_out(N_SWITCHES, meta_agent, env, HORIZON, quiet)
#             meta_caches.append(meta_c)
#             meta_regrets[i, j] = meta_r[-1]
#             meta_trick_r, meta_trick_c = meta_rolls_out(N_SWITCHES, meta_trick_agent, env, HORIZON, quiet)
#             meta_trick_caches.append(meta_trick_c)
#             meta_trick_regrets[i, j] = meta_trick_r[-1]
#             exp3_r, exp3_c = meta_rolls_out(N_SWITCHES, exp3_agent, env, HORIZON, quiet)
#             exp3_caches.append(exp3_c)
#             exp3_regrets[i, j] = exp3_r[-1]
#             moss_r, moss_c = meta_rolls_out(N_SWITCHES, moss_agent, env, HORIZON, quiet)
#             moss_caches.append(moss_c)
#             moss_regrets[i, j] = moss_r[-1]
#             exp3_reset_r, exp3_reset_c = meta_rolls_out(N_SWITCHES, exp3_reset_agent, env, HORIZON, quiet)
#             exp3_reset_caches.append(exp3_reset_c)
#             exp3_reset_regrets[i, j] = exp3_reset_r[-1]

#     X = n_experts_list
#     gap = kwargs['gap_constrain']
#     title = f'Regret: {N_BANDITS} arms, horizon {HORIZON}, {N_SWITCHES} switches, gap = {gap} and subgroup size {OPT_SIZE}'
#     xlabel, ylabel = 'Number of Expert', 'Regret'
#     plot(X, meta_regrets, exp3_regrets, moss_regrets, exp3_reset_regrets, meta_trick_regrets, title, xlabel, ylabel)
#     return (X, meta_regrets, exp3_regrets, moss_regrets, exp3_reset_regrets, meta_trick_regrets, title, xlabel, ylabel)
