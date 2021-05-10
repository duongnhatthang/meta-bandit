import numpy as np
from scipy.special import softmax


class MOSS:
    def __init__(self, n_arms, horizon):
        self.n_arms = n_arms
        self.horizon = horizon

    def get_action(self, obs):
        mu = obs[::2]
        T = obs[1::2]
        T[T == 0] = 1e-6
        log_plus = np.log(np.maximum(1, self.horizon / (self.n_arms * T)))
        index = mu + np.sqrt(4 * log_plus / T)
        return np.argmax(index)

    def reset(self):
        pass


class ExpertMOSS(MOSS):
    """
    Assign the index of all arms outside of expert_subset as min_index
    """

    def __init__(self, n_arms, horizon, expert_subset, min_index=-1000):
        super().__init__(n_arms, horizon)
        self.expert_subset = expert_subset
        self.min_index = min_index

    def get_action(self, obs):
        mu = obs[::2]
        T = obs[1::2]
        T[T == 0] = 1e-6
        log_plus = np.log(np.maximum(1, self.horizon / (self.n_arms * T)))
        index = mu + np.sqrt(4 * log_plus / T)
        mask = np.ones((self.n_arms,))
        mask[self.expert_subset] = 0
        mask = mask.astype(bool)
        index[mask] = self.min_index
        return np.argmax(index)


class PhaseElim:
    def __init__(self, n_arms, horizon, C=1, min_index=-1000):
        self.n_arms = n_arms
        self.horizon = horizon
        self.C = C
        self.reset()
        self.min_index = min_index
        if self.n_arms*self._get_ml() > self.horizon:
            print(f"WARNING (Phased Elimination): phase 1 duration ({self.n_arms*self._get_ml()}) is larger than the horizon ({self.horizon}) => increase horizon and/or change n_arms.")

    def reset(self):
        self.ml_counter = -1
        self.cur_l = 1  # phase counter
        self.A_l = np.arange(self.n_arms)
        self.cur_mu = np.zeros((self.n_arms,))  # Tracking mu in one phase
        self.cur_phase_actions = np.repeat(
            self.A_l, self._get_ml()
        )  # get 'ml' observations from each arm in the self.A_l

    def _get_ml(self):
        return round(
            self.C
            * 2 ** (2 * self.cur_l)
            * np.log(max(np.exp(1), self.n_arms * self.horizon * 2 ** (-2 * self.cur_l)))
        )

    def _eliminate(self):
        if self.A_l.shape[0] == 1:
            return
        max_mu = np.max(self.cur_mu)
        eliminate_arm_index = np.where(self.cur_mu + 2 ** (-self.cur_l) < max_mu)[0]
        self.A_l = np.setdiff1d(
            self.A_l, eliminate_arm_index
        )  # yields the elements in `self.A_l` that are NOT in `eliminate_arm_index`
        self.cur_mu = np.zeros((self.n_arms,))
        self.cur_mu[eliminate_arm_index] = self.min_index

    def get_action(self, obs):
        if self.ml_counter == self._get_ml() * self.A_l.shape[0] - 1:
            # Reset statistics when starting a new phase
            self.ml_counter = -1
            self.cur_l += 1
            self._eliminate()
            self.cur_phase_actions = np.repeat(self.A_l, self._get_ml())
        self.ml_counter += 1
        return self.cur_phase_actions[self.ml_counter]

    def update(self, action, reward):
        self.cur_mu[action] += reward / self._get_ml()


class EE:
    """
    Exploration-Exploitation algorithm:
    - Run EXR with probability delta_n
    - Aggregate surviving arms until it contains |S| arms (size of optimal subset)
    - => Then only run EXT
    """

    def __init__(self, n_arms, horizon, n_tasks, expert_subsets, C=1, min_index=-1000):
        self.min_index = min_index
        self.n_arms = n_arms
        self.horizon = horizon
        self.expert_subsets = expert_subsets
        self.n_experts = self.expert_subsets.shape[0]
        self.n_tasks = n_tasks
        self.PE_algo = PhaseElim(n_arms, horizon, C, min_index)
        self.reset()
        self.MOSS_algo = MOSS(self.n_arms, self.horizon)
        self.subset_size = expert_subsets[0].shape[0]
        self.C1 = np.sqrt(horizon * self.subset_size)
        self.C2 = np.sqrt(horizon * n_arms)
        self.C3 = horizon
        self.predicted_opt_arms = []
        self.cur_task = 0
        self._set_is_explore()

    def reset(self):
        self.PE_algo.reset()

    def _get_delta_n(self):
        if self.n_tasks == self.cur_task:
            return 1
        numerator = self.C3 - self.C1
        denominator = (self.C2 - self.C1) * 2 * (self.n_tasks - self.cur_task - 1)
        return np.sqrt(numerator / max(1e-6, denominator))

    def _set_is_explore(self):
        p = self._get_delta_n()
        p = min(p, 1)  # TODO: prob bug here
        self.is_explore = bool(np.random.choice(2, p=[1 - p, p]))

    def get_action(self, obs):  # get action for each rolls-out step
        if self.is_explore and len(self.predicted_opt_arms) < self.subset_size:
            return self.PE_algo.get_action(obs)
        else:
            return self.MOSS_algo.get_action(obs)

    def eps_end_update(self, obs):  # update the tracking_stats after each rolls-out
        if self.is_explore and len(self.predicted_opt_arms) < self.subset_size:
            arms_found = self.PE_algo.A_l
            if arms_found.shape[0] == 1:
                if arms_found[0] not in self.predicted_opt_arms:
                    self.predicted_opt_arms.append(arms_found[0])
            else:
                self.predicted_opt_arms += arms_found.tolist()
                self.predicted_opt_arms = list(set(self.predicted_opt_arms))
            self.MOSS_algo = ExpertMOSS(self.n_arms, self.horizon, self.predicted_opt_arms, self.min_index)
        self.cur_task += 1
        self._set_is_explore()

    def update(self, action, reward):
        if self.is_explore and len(self.predicted_opt_arms) < self.subset_size:
            self.PE_algo.update(action, reward)


class PMML_EWA:
    """
    Tracking the statistic of each EXT experts and 1 EXR expert => EWA.
     - If EXT expert contain PE survival arms => 0 cost, else => (C3-C1)/P_{EXR} cost
     - EXR expert => (C2-C1)/P_{EXR}
     - Only update statistic at EXR round
    """

    def __init__(self, n_arms, horizon, n_tasks, expert_subsets, C=1, min_index=-1000):
        self.n_arms = n_arms
        self.horizon = horizon
        self.n_tasks = n_tasks
        self.expert_subsets = expert_subsets
        self.subset_size = expert_subsets[0].shape[0]
        self.n_experts = self.expert_subsets.shape[0]
        self.min_index = min_index
        self.C1 = np.sqrt(horizon * self.subset_size)
        self.C2 = np.sqrt(horizon * n_arms)
        self.C3 = horizon
        self.learning_rate = self._default_learning_rate()
        self.tracking_stats = np.zeros((self.n_experts + 1,))  # Last expert is EXR
        self.tracking_stats[-1] = 1
        self.PE_algo = PhaseElim(n_arms, horizon, C, min_index)
        self.reset()
        self.delta_n = self._get_delta_n()
        assert (
            self.delta_n <= 1 and self.delta_n >= 0
        ), f" self.delta_n ({self.delta_n}) is not in the range [0,1]. Reduce N_EXPERT, HORIZON or increase n_tasks."
        self._select_expert()
        assert (
            self.C1 <= self.C2 and self.C2 <= self.C3
        ), f"C1 ({self.C1}) < C2 ({self.C2}) < C3 ({self.C3}) not satisfied."

    def reset(self):
        self.PE_algo.reset()

    def _default_learning_rate(self):
        return 1

    def _get_delta_n(self):
        return (self.C3 * np.log(self.n_experts) / (self.C2 * self.n_tasks)) ** (1 / 2)

    def _select_expert(self):
        # EWA algorithm, max softmax trick
        tmp = self.learning_rate * self.tracking_stats
        tmp -= tmp.max()
        Q_n = softmax(tmp)
        P_n = np.zeros((self.n_experts + 1,))
        P_n[-1] = self.delta_n
        self.P_n = P_n + (1 - self.delta_n) * Q_n  # Expert distribution to select/sample from
        if (self.P_n == 0).any():  # fix 0 probability
            self.P_n[self.P_n == 0] = 1e-6
            self.P_n /= np.sum(self.P_n)
        self.cur_subset_index = np.random.choice(self.n_experts + 1, p=self.P_n)
        if self.cur_subset_index < self.n_experts:  # EXT: exploit
            cur_subset = self.expert_subsets[self.cur_subset_index]
            self.cur_algo = ExpertMOSS(self.n_arms, self.horizon, cur_subset)
        else:  # EXR: explore
            self.cur_algo = self.PE_algo

    def get_action(self, obs):  # get action for each rolls-out step
        return self.cur_algo.get_action(obs)

    def eps_end_update(self, obs):  # update the tracking_stats after each rolls-out
        if self.cur_subset_index == self.n_experts:  # EXR: explore
            self._update_tracking_stats(obs)
        self._select_expert()

    def _get_tilda_c_n(self):
        tilda_c_n = np.zeros((self.n_experts + 1,))
        tilda_c_n[-1] = self.C2
        surviving_arms = self.PE_algo.A_l
        experts_contains_surviving_arms = []
        for i, e in enumerate(self.expert_subsets):
            tmp = np.intersect1d(surviving_arms, e)
            if len(tmp) > 0:
                experts_contains_surviving_arms.append(i)
        tilda_c_n[np.arange(self.n_experts).astype(int)] = self.C3
        tilda_c_n[experts_contains_surviving_arms] = self.C1
        tilda_c_n -= self.C1
        tilda_c_n /= self.P_n[-1]
        return tilda_c_n

    def _get_loss_vector(self):
        tilda_c_n = self._get_tilda_c_n()
        l_n = self.delta_n * tilda_c_n / self.C3
        return l_n

    def _update_tracking_stats(self, obs):
        l_n = self._get_loss_vector()
        self.tracking_stats += 1 - l_n

    def update(self, action, reward):
        if self.cur_subset_index == self.n_experts:  # EXR: exploration
            self.PE_algo.update(action, reward)


class PMML(PMML_EWA):
    """
    Remove all experts not contain the surviving arms returned by Phase Elimination
    """

    def __init__(self, n_arms, horizon, n_tasks, expert_subsets, C=1, min_index=-1000):
        super().__init__(n_arms, horizon, n_tasks, expert_subsets, C, min_index)
        self.surviving_experts = np.arange(self.n_experts)

    def _update_tracking_stats(self, obs):
        l_n = self._get_loss_vector()
        surviving_experts = np.where(l_n == 0)[0]
        self.surviving_experts = np.intersect1d(self.surviving_experts, surviving_experts)
        if self.surviving_experts.shape[0] == 1:  # stop Exploration after finding the correct expert
            cur_subset = self.expert_subsets[self.surviving_experts[0]]
            self.cur_algo = ExpertMOSS(self.n_arms, self.horizon, cur_subset)
        else:
            temp = np.ones_like(self.tracking_stats) * self.min_index
            temp[self.surviving_experts] = self.tracking_stats[self.surviving_experts]
            temp[-1] = self.tracking_stats[-1]  # EXR expert statistic
            self.tracking_stats = temp
            self.tracking_stats += 1 - l_n

    def eps_end_update(self, obs):  # update the tracking_stats after each rolls-out
        self._update_tracking_stats(obs)
        if self.surviving_experts.shape[0] > 1:  # Only EWA to select expert if there are more than 1 surviving
            self._select_expert()


class GML:
    """
    Greedy algorithm for bandit meta-learning
    """

    def __init__(self, n_arms, horizon, n_tasks, expert_subsets, C=1, min_index=-1000):
        self.n_arms = n_arms
        self.horizon = horizon
        self.n_tasks = n_tasks
        self.expert_subsets = expert_subsets
        self.subset_size = expert_subsets[0].shape[0]
        self.n_experts = self.expert_subsets.shape[0]
        self.min_index = min_index
        self.B_TK = np.sqrt(horizon * self.n_arms * np.log(self.n_arms)) 
        self.tracking_stats = np.zeros((n_tasks,n_arms))
        self.EXT_set = None
        self.is_explore = None
        self.cur_task = 0
        self.PE_algo = PhaseElim(n_arms, horizon, C, min_index)
        self.reset()
        self.select_alg()

    def reset(self):
        self.PE_algo.reset()

    def find_EXT_set(self):
        """
        Greedy algorithm to for Hitting Set Problem.
        Return set 's' in the paper
        """
        M = np.nonzero(np.sum(self.tracking_stats, axis = 0))[0].shape[0] # The number of arms returned by past PE
        assert M > 0, "Running EXT in the first task"
        self.EXT_set = []
        mask = np.zeros((self.n_tasks,), dtype=bool)
        EXR_idxs = np.nonzero(np.sum(self.tracking_stats, axis = 1))
        mask[EXR_idxs] = True
        for i in range(M):
            tmp = np.sum(self.tracking_stats[mask], axis = 0) # shape = (K,)
            max_arm_idx = np.argmax(tmp)
            self.EXT_set.append(max_arm_idx)
            task_idxs = np.nonzero(self.tracking_stats[:, max_arm_idx]) # make sure axis 0 is correct => correct idxs
            mask[task_idxs] = False
            if np.sum(mask)==0: # Covered all tasks
                break

    def get_EXR_prob(self):
        if self.cur_task == 0 or self.cur_task > self.n_tasks - 2 : # force EXR
            return 1
        self.find_EXT_set()
        B_Ts = np.sqrt(self.horizon * len(self.EXT_set))
        # G_{n+1}, the extra "-1" is because cur_task count from 0
        G = np.sqrt(2*(self.B_TK-B_Ts)*(self.horizon-B_Ts)*(self.n_tasks-self.cur_task-2))
        p = (self.horizon-B_Ts) / (self.horizon-B_Ts+G)
        return p

    def set_is_explore(self):
        p = self.get_EXR_prob()
        self.is_explore = bool(np.random.choice(2, p=[1 - p, p]))

    def select_alg(self):
        self.set_is_explore()
        if self.is_explore:
            self.cur_algo = self.PE_algo
        else:
            self.cur_algo = ExpertMOSS(self.n_arms, self.horizon, self.EXT_set)

    def get_action(self, obs):  # get action for each rolls-out step
        return self.cur_algo.get_action(obs)

    def eps_end_update(self, obs):  # update the tracking_stats after each rolls-out
        if self.is_explore:
            self.update_tracking_stats(obs)
        self.select_alg()
        self.cur_task += 1

    def update_tracking_stats(self, obs):
        surviving_arms = self.PE_algo.A_l
        self.tracking_stats[self.cur_task, surviving_arms] = 1

    def update(self, action, reward):
        if self.is_explore:
            self.PE_algo.update(action, reward)
