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
        super().__init__(n_bandits, n_tasks)
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
        done = False
        self.cur_step += 1
        return copy.deepcopy(self.counter), reward, done, self.caches

class MetaBernoulli(Bernoulli):    
    """
        All task's optimal arms is in a sub-group
    """
    def __init__(self, n_bandits, opt_size, n_tasks, n_experts, **kwargs):
        if n_experts is not None:
            assert n_experts > 0, f"n_experts ({n_experts}) must be larger than 0."
        self.r_dist = np.full((n_tasks,n_bandits), 1)
        self.n_tasks = n_tasks
        self.n_bandits = n_bandits
        self.action_space = spaces.Discrete(self.n_bandits)
        self.observation_space = spaces.Discrete(2*n_bandits)
        self.n_experts = n_experts
        self.opt_size = opt_size
        self.gap_constrain = kwargs['gap_constrain']

        self.opt_indices = np.arange(n_bandits)
        np.random.shuffle(self.opt_indices)
        self.sub_opt_indices = self.opt_indices[opt_size:]
        self.opt_indices = np.sort(self.opt_indices[:opt_size]) # The indices of the optimal sub-group
        low = 1e-6
        if self.gap_constrain is not None:
            low = self.gap_constrain
        while True:
            self.p_dist = np.zeros((n_tasks,n_bandits))
            opt_values = np.random.uniform(low = low, size = (n_tasks,))
            while True:
                opt_indices = np.random.choice(self.opt_indices, size = (n_tasks,))
                if len(list(set(opt_indices.tolist()))) == opt_size:
                    break
            temp = np.random.uniform(high = opt_values-low, size=(n_bandits, n_tasks))
            self.p_dist = temp.T
            self.p_dist[np.arange(n_tasks), opt_indices] = opt_values
            if np.max(opt_values)>0:
                break
        self.reset_task(0)

        if self.n_experts is None: #All combinations
            tmp = list(combinations(np.arange(self.n_bandits), self.opt_size))
            self.n_experts = len(tmp)
            self.expert_subgroups = np.asarray(tmp)
        else:
            self.expert_subgroups = np.zeros((self.n_experts, self.opt_size))
            for i in range(self.n_experts):
                while True: # Sample a new subgroup, if not appeared before, added to self.expert_subgroups
                    tmp = np.random.choice(self.n_bandits, self.opt_size, replace=False)
                    if any((self.expert_subgroups[:i]==tmp).all(1)) == False:
                        self.expert_subgroups[i] = tmp
                        break
            self.expert_subgroups = self.expert_subgroups.astype(int)
            # Check if The Optimal subgroup (self.opt_indices) is inside the self.expert_subgroups
            if any((self.expert_subgroups[:]==self.opt_indices).all(1)) == False:
                i = np.random.randint(n_experts)
                self.expert_subgroups[i] = self.opt_indices
        print(f'Optimal expert index = {np.where((self.expert_subgroups[:]==self.opt_indices).all(1))[0][0]}')
