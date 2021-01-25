import algos
import bandit
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations
import utils
import algos
import bandit
from tqdm import tqdm, trange

plt.rcParams['figure.figsize'] = [8, 4]
plt.rcParams['figure.dpi'] = 100 # 200 e.g. is really fine, but slower

DS_NAME = None #None for synthesize, or {"LastFM"}
N_SWITCHES = 3000 # LastFM: 1892
N_BANDITS = 3 # LastFM: 17632
OPT_SIZE = 2
HORIZON = 100 # LastFM: 10*17632
max_n_expert = len(list(combinations(np.arange(N_BANDITS),OPT_SIZE)))
N_EXPERT = None #All possible combinations
# N_EXPERT = 10
if N_EXPERT is not None:
    print(f'N_EXPERT = {N_EXPERT}')
    assert N_EXPERT<=max_n_expert, f"The number of expert ({N_EXPERT}) must be smaller than the maximum combination ({max_n_expert})"
assert N_BANDITS<=HORIZON, f"The number of arm ({N_BANDITS}) must be smaller than the horizon ({HORIZON})"
N_EXPS = 100 #Repeat experiments
kwargs = {'switches_cache_step':100}

#MetaAlg params
N_UNBIASED_OBS = 1
ALG_NAME = 'ExpertMOSS'

quiet = True
horizon_list = np.array([1, 100, 300])
env = bandit.MetaBernoulli(n_bandits=N_BANDITS, opt_size=OPT_SIZE, n_tasks=N_SWITCHES+1, 
                       n_experts=N_EXPERT, ds_name=DS_NAME, **kwargs)
exp3_caches_h, exp3_reset_caches_h, moss_caches_h, meta_caches_h = [], [], [], []
exp3_reset_regrets_h = np.zeros((N_EXPS, horizon_list.shape[0]))
meta_regrets_h = np.zeros((N_EXPS, horizon_list.shape[0]))
exp3_regrets_h = np.zeros((N_EXPS, horizon_list.shape[0]))
moss_regrets_h = np.zeros((N_EXPS, horizon_list.shape[0]))
exp3_kwargs = {'n_switches':N_SWITCHES}
for i in trange(N_EXPS):
    for j, h in enumerate(horizon_list):
        exp3_agent = algos.Exp3(n_bandits=N_BANDITS, horizon=h, **exp3_kwargs)
        exp3_reset_agent = algos.Exp3(n_bandits=N_BANDITS, horizon=h, is_reset=True)
        moss_agent = algos.MOSS(n_bandits=N_BANDITS, horizon=h)
        meta_agent = algos.MetaAlg(n_bandits=N_BANDITS, horizon=h, n_switches=N_SWITCHES, 
                          n_unbiased_obs=N_UNBIASED_OBS, alg_name=ALG_NAME, 
                          expert_subgroups=env.expert_subgroups)
        meta_r, meta_c = utils.meta_rolls_out(N_SWITCHES, meta_agent, env, h, quiet)
        meta_caches_h.append(meta_c)
        meta_regrets_h[i, j] = meta_r[-1]/h
        exp3_r, exp3_c = utils.meta_rolls_out(N_SWITCHES, exp3_agent, env, h, quiet)
        exp3_caches_h.append(exp3_c)
        exp3_regrets_h[i, j] = exp3_r[-1]/h
        moss_r, moss_c = utils.meta_rolls_out(N_SWITCHES, moss_agent, env, h, quiet)
        moss_caches_h.append(moss_c)
        moss_regrets_h[i, j] = moss_r[-1]/h
        exp3_reset_r, exp3_reset_c = utils.meta_rolls_out(N_SWITCHES, exp3_reset_agent, env, h, quiet)
        exp3_reset_caches_h.append(exp3_reset_c)
        exp3_reset_regrets_h[i, j] = exp3_reset_r[-1]/h

# X = horizon_list
# if N_EXPERT is None:
#     N_EXPERT = env.n_experts
# title = f'Regret: {N_BANDITS} arms, {N_SWITCHES} switches, {N_EXPERT} experts and subgroup size {OPT_SIZE}'
# xlabel, ylabel = 'Horizon', 'Average Regret per Step'

with open('meta_regrets_h.npy', 'wb') as f:
    np.save(f, meta_regrets_h)
with open('exp3_regrets_h.npy', 'wb') as f:
    np.save(f, exp3_regrets_h)
with open('moss_regrets_h.npy', 'wb') as f:
    np.save(f, moss_regrets_h)
with open('exp3_reset_regrets_h.npy', 'wb') as f:
    np.save(f, exp3_reset_regrets_h)
