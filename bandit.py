import numpy as np
import copy
from gym import spaces
from gym import Env
from itertools import combinations
from collections import Counter
import pandas as pd

class NoStatBernoulli(Env):
    """
    control agent to choose an arm
    tasks (aka goals) maximize cumulative reward for different bandit setting
    Basic bandit setting, independent reward between arms
    Observation: None
    Action: 0-n (n is # of arms)
    Reward: {0,1} bernoulli distribution of chosen arm
    """

    def __init__(self, n_bandits, n_tasks):
        while True:
            self.p_dist = np.random.uniform(size=(n_tasks,n_bandits))
            if np.max(self.p_dist)>0:
                break
        self.r_dist = np.full((n_tasks,n_bandits), 1)
        self.n_tasks = n_tasks
        self.n_bandits = n_bandits
        self.action_space = spaces.Discrete(self.n_bandits)
        self.observation_space = spaces.Discrete(1)
        self.reset_task(0)

    def step(self, action):
        """
        Deterministic. When it's stochastic, used self._reward_func and self._trans_prob_func
        """
        action = int(np.rint(action))
        reward = np.random.choice(2,p=[1-self._p[action],self._p[action]])*self._r[action]
        done = False
        ob = self._get_obs()
        self.cur_step += 1
        return ob, reward, done, dict()

    def _get_obs(self):
        return -1

    def reset_task(self, idx):
        ''' change current env to idx '''
        self.cur_index = int(idx)
        self._p = self.p_dist[self.cur_index]
        self._r = self.r_dist[self.cur_index]
        return self.reset()

    def reset(self):
        self.cur_step = -1
        return self._get_obs()

    def viewer_setup(self):
        print('no viewer')
        pass

    def render(self):
        print('no render')
        pass
        
class Bernoulli(NoStatBernoulli):    
    """
    Observation: return statistic (avg_0, #_chosen_0, avg_1, ...)
    """
    def __init__(self, n_bandits, n_tasks):
        super().__init__(self, n_bandits, n_tasks)
        self.observation_space = spaces.Discrete(2*n_bandits)

    def _get_obs(self):
        return copy.deepcopy(self.counter)

    def reset(self):
        self.caches = [] # For debug only
        self.total_reward = 0
        self.counter = np.zeros((2*self.n_bandits,))
        self.cur_step = -1
        return self._get_obs()

    def step(self, action):
        """
        Deterministic gridworld. When it's stochastic, used self._reward_func and self._trans_prob_func
        """
        action = int(np.rint(action))
        reward = np.random.choice(2,p=[1-self._p[action],self._p[action]])*self._r[action]
        self.total_reward+=reward
        choice_avg = self.counter[2*action]
        choice_no_pick = self.counter[2*action+1]
        self.counter[2*action] = (choice_avg*choice_no_pick + reward)/(choice_no_pick + 1)
        self.counter[2*action+1] = choice_no_pick + 1
        self.caches.append(action)
        done = False
        self.cur_step += 1
        return copy.deepcopy(self.counter), reward, done, self.caches

class MetaBernoulli(Bernoulli):    
    """
        All task's optimal arms is in a sub-group
    """
    def __init__(self, n_bandits, opt_size, n_tasks, n_experts, ds_name = None):
        assert n_experts > 0, f"n_experts ({n_experts}) must be larger than 0."
        self.r_dist = np.full((n_tasks,n_bandits), 1)
        self.n_tasks = n_tasks
        self.n_bandits = n_bandits
        self.action_space = spaces.Discrete(self.n_bandits)
        self.observation_space = spaces.Discrete(2*n_bandits)
        self.n_experts = n_experts
        self.opt_size = opt_size
        
        if ds_name is None: #Synthesize dataset
            self.opt_indices = np.arange(n_bandits)
            np.random.shuffle(self.opt_indices)
            self.sub_opt_indices = self.opt_indices[opt_size:]
            self.opt_indices = np.sort(self.opt_indices[:opt_size]) # The indices of the optimal sub-group
            while True:
                self.p_dist = np.zeros((n_tasks,n_bandits))
                self.p_dist[:, self.opt_indices] = np.random.uniform(size=(n_tasks,opt_size))
                opt_values = self.p_dist.max(axis=1)

                temp = np.random.uniform(high = opt_values, size=(n_bandits - opt_size, n_tasks))
                self.p_dist[:, self.sub_opt_indices] = temp.T
                if np.max(opt_values)>0:
                    break
        elif ds_name == "LastFM":
            self._load_lastfm()
        else:
            assert False, f"{ds_name} is not implemented."

        self.reset_task(0)
        all_subgroups = np.array(list(combinations(np.arange(n_bandits),opt_size)))
        idxs = (all_subgroups == self.opt_indices).all(1)
        opt_idx = np.where(idxs)[0][0]

        tmp = np.random.choice(all_subgroups.shape[0], n_experts, replace=False)
        self.expert_subgroups = all_subgroups[tmp]
        if opt_idx not in tmp:
            i = np.random.randint(n_experts)
            self.expert_subgroups[i] = self.opt_indices
            
    def _load_lastfm(self):
        artists_df = pd.read_csv('./raw_ds/hetrec2011-lastfm-2k/artists.dat',sep='\t')
        user_artists_df = pd.read_csv('./raw_ds/hetrec2011-lastfm-2k/user_artists.dat',sep='\t')
        
        unique_userID = np.unique(user_artists_df['userID'].values)
        unique_artistID = np.unique(artists_df['id'].values)
        assert unique_artistID.shape[0] == self.n_bandits, f"unique_artistID ({unique_artistID}) is not equal the number of arm ({self.n_bandits})"
        assert unique_userID.shape[0] == self.n_tasks, f"unique_userID ({unique_userID}) is not equal the number of task ({self.n_tasks})"
        self.p_dist = np.zeros((unique_userID.shape[0], unique_artistID.shape[0]))
        for _, row in user_artists_df.iterrows():
            u_idx = np.where(unique_userID==row['userID'])[0][0]
            a_idx = np.where(unique_artistID==row['artistID'])[0][0]
            self.p_dist[u_idx, a_idx] = row['weight']
        for i in range(self.p_dist.shape[0]):
            self.p_dist[i] /= self.p_dist[i].sum()

        c = Counter(np.argmax(self.p_dist, axis=1))

        self.opt_indices = np.zeros((self.opt_size,)).astype(int)
        for i,tmp in enumerate(c.most_common()):
            if i == self.opt_size:
                break
            else:
                self.opt_indices[i] = tmp[0]