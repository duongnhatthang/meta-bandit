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

# plt.rcParams['figure.figsize'] = [8, 4]
# plt.rcParams['figure.dpi'] = 100 # 200 e.g. is really fine, but slower

DS_NAME = None #None for synthesize, or {"LastFM"}
N_SWITCHES = 10000 # LastFM: 1892
N_BANDITS = 3 # LastFM: 17632
OPT_SIZE = 2
HORIZON = 75 # LastFM: 10*17632
max_n_expert = len(list(combinations(np.arange(N_BANDITS),OPT_SIZE)))
N_EXPERT = None #All possible combinations
# N_EXPERT = 10
if N_EXPERT is not None:
    print(f'N_EXPERT = {N_EXPERT}')
    assert N_EXPERT<=max_n_expert, f"The number of expert ({N_EXPERT}) must be smaller than the maximum combination ({max_n_expert})"
assert N_BANDITS<=HORIZON, f"The number of arm ({N_BANDITS}) must be smaller than the horizon ({HORIZON})"
N_EXPS = 10 #Repeat experiments
kwargs = {'switches_cache_step':100}

#MetaAlg params
N_UNBIASED_OBS = 1
ALG_NAME = 'ExpertMOSS'

(X, meta_regrets, exp3_regrets, moss_regrets, exp3_reset_regrets, meta_trick_regrets, title, xlabel, ylabel) = utils.switches_exp(N_EXPS, N_SWITCHES, N_BANDITS, HORIZON, 
                   N_UNBIASED_OBS, ALG_NAME, OPT_SIZE, N_EXPERT, DS_NAME, **kwargs)

with open('meta_regrets.npy', 'wb') as f:
    np.save(f, meta_regrets)
with open('meta_trick_regrets.npy', 'wb') as f:
    np.save(f, meta_trick_regrets)
with open('exp3_regrets.npy', 'wb') as f:
    np.save(f, exp3_regrets)
with open('moss_regrets.npy', 'wb') as f:
    np.save(f, moss_regrets)
with open('exp3_reset_regrets.npy', 'wb') as f:
    np.save(f, exp3_reset_regrets)
