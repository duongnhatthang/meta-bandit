import copy
from itertools import combinations
from scipy.special import comb
import numpy as np
from gym import Env, spaces


class NoStatBernoulli(Env):
    """
    Basic bandit setting, independent reward between arms
    Observation: None
    Action: [0,n) (n is # of arms)
    Reward: {0,1} bernoulli distribution of chosen arm
    """

    def __init__(self, n_arms, n_tasks):
        while True:
            self.p_dist = np.random.uniform(size=(n_tasks, n_arms))
            if np.max(self.p_dist) > 0:
                break
        self.r_dist = np.full((n_tasks, n_arms), 1)
        self.n_tasks = n_tasks
        self.n_arms = n_arms
        self.action_space = spaces.Discrete(self.n_arms)
        self.observation_space = spaces.Discrete(1)
        self.reset_task(0)

    def step(self, action):
        action = int(np.rint(action))
        reward = np.random.choice(2, p=[1 - self._p[action], self._p[action]]) * self._r[action]
        done = False
        ob = self._get_obs()
        self.cur_step += 1
        return ob, reward, done, dict()

    def _get_obs(self):
        return -1

    def reset_task(self, idx):
        """ change current env to idx """
        self.cur_task = int(idx)
        self._p = self.p_dist[self.cur_task]
        self._r = self.r_dist[self.cur_task]
        return self.reset()

    def reset(self):
        self.cur_step = -1
        return self._get_obs()

    def viewer_setup(self):
        print("no viewer")
        pass

    def render(self):
        print("no render")
        pass


class Bernoulli(NoStatBernoulli):
    """
    Observation: return statistic (avg_0, #_chosen_0, avg_1, ...)
    """

    def __init__(self, n_arms, n_tasks):
        super().__init__(n_arms, n_tasks)
        self.observation_space = spaces.Discrete(2 * n_arms)

    def _get_obs(self):
        return copy.deepcopy(self.counter)

    def reset(self):
        self.caches = []  # For debug only
        self.total_reward = 0
        self.counter = np.zeros((2 * self.n_arms,))
        self.cur_step = -1
        return self._get_obs()

    def step(self, action):
        action = int(np.rint(action))
        reward = np.random.choice(2, p=[1 - self._p[action], self._p[action]]) * self._r[action]
        self.total_reward += reward
        choice_avg = self.counter[2 * action]
        choice_no_pick = self.counter[2 * action + 1]
        self.counter[2 * action] = (choice_avg * choice_no_pick + reward) / (choice_no_pick + 1)
        self.counter[2 * action + 1] = choice_no_pick + 1
        done = False
        self.cur_step += 1
        return copy.deepcopy(self.counter), reward, done, self.caches


class MetaBernoulli(Bernoulli):
    """
    All task's optimal arms is in a sub-group
    """

    def __init__(self, n_arms, opt_size, n_tasks, **kwargs):
        self.r_dist = np.full((n_tasks, n_arms), 1)
        self.n_tasks = n_tasks
        self.n_arms = n_arms
        self.action_space = spaces.Discrete(self.n_arms)
        self.observation_space = spaces.Discrete(2 * n_arms)
        self.opt_size = opt_size
        self.gap_constrain = None
#         self.gap_constrain = kwargs["gap_constrain"]
        self.n_experts = comb(self.n_arms, self.opt_size)

        self.opt_indices = np.arange(n_arms)
        np.random.shuffle(self.opt_indices)
        self.sub_opt_indices = self.opt_indices[opt_size:]
        self.opt_indices = np.sort(self.opt_indices[:opt_size])  # The indices of the optimal sub-group
        low = 1e-6
        if self.gap_constrain is not None:
            low = self.gap_constrain
        while True:
            self.p_dist = np.zeros((n_tasks, n_arms))
            opt_values = np.random.uniform(low=low, size=(n_tasks,))
            while True:
                opt_indices = np.random.choice(self.opt_indices, size=(n_tasks,))
                if len(list(set(opt_indices.tolist()))) == opt_size:
                    break
            temp = np.random.uniform(high=opt_values - low, size=(n_arms, n_tasks))
            self.p_dist = temp.T
            self.p_dist[np.arange(n_tasks), opt_indices] = opt_values
            if np.max(opt_values) > 0:
                break
        self.reset_task(0)
        print(f"opt_indices = {self.opt_indices}")


class OriginalAdvMetaBernoulli(MetaBernoulli):

    def __init__(self, n_arms, opt_size, n_tasks, horizon, **kwargs):
        super().__init__(n_arms, opt_size, n_tasks, **kwargs)
        self.horizon = horizon
        self.B_TK = np.sqrt(horizon * n_arms * np.log(n_arms))
        self.B_TM = np.sqrt(horizon * opt_size)

    def generate_next_task(self, EXT_set):
        if self.cur_task > self.n_tasks - 2: # Final task
            return
        if EXT_set is None: # First round
            opt_arm = np.random.choice(self.opt_indices)
        else:
            correct_EXT_set = np.intersect1d(self.opt_indices, EXT_set)
            G = np.sqrt(2*(self.B_TK-self.B_TM)*(self.horizon-self.B_TM)*(self.n_tasks-self.cur_task-2))
            q = (self.B_TK-self.B_TM) / (self.horizon-self.B_TM+G) # probability that next task optimal arm is NOT in correct_EXT_set
            self.is_out_of_set = bool(np.random.choice(2, p=[1 - q, q]))
            inv_correct_EXT_set = np.setdiff1d(self.opt_indices, correct_EXT_set)
            if len(inv_correct_EXT_set)==0:
                opt_arm = np.random.choice(correct_EXT_set)
            else:
                if self.is_out_of_set is True or len(correct_EXT_set)==0:
                    opt_arm = np.random.choice(inv_correct_EXT_set)
                else:
                    opt_arm = np.random.choice(correct_EXT_set)

        low = 1e-6
        if self.gap_constrain is not None:
            low = self.gap_constrain
        opt_values = np.random.uniform(low=low)
        next_p = np.random.uniform(high=opt_values - low, size=(self.n_arms, ))
        next_p[opt_arm] = opt_values
        self.p_dist[self.cur_task+1] = next_p

class AdvMetaBernoulli(OriginalAdvMetaBernoulli):
    """
    Using internal EXT_set instead of looking at the agent's EXT_set
    """
    def __init__(self, n_arms, opt_size, n_tasks, horizon, **kwargs):
        super().__init__(n_arms, opt_size, n_tasks, horizon, **kwargs)
        self.EXT_set = []
        for i in range(self.n_tasks-1):
            self.reset_task(i)
            opt_arm_idx = np.argmax(self.p_dist[i])
            if opt_arm_idx not in self.EXT_set:
#                 G = np.sqrt(2*(self.B_TK-self.B_TM)*(self.horizon-self.B_TM)*(self.n_tasks-self.cur_task-2))
#                 p = (self.horizon-self.B_TM) / (self.horizon-self.B_TM+G) # probability that the agent will EXR and find this new optimal arm
                # Commented aboved are part 3.1 (gap condition satisfied). Below are the general strategy
                p = np.sqrt((self.opt_size*self.horizon)/(self.n_tasks*self.B_TK))
                does_find_new_arm = bool(np.random.choice(2, p=[1 - p, p]))
                if does_find_new_arm is True:
                    self.EXT_set.append(opt_arm_idx)
            super().generate_next_task(self.EXT_set)
        self.reset_task(0)

    def generate_next_task(self, EXT_set): # Placeholder
        pass
