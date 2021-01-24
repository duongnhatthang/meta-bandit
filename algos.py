import numpy as np
import random
import copy
from scipy.special import softmax

class Random:
    def __init__(self, n_bandits):
        self.n_bandits = n_bandits
        
    def get_action(self,obs):
        """
            Observation: return statistic (avg_0, #_chosen_0, avg_1, ...)
        """
        return np.random.randint(self.n_bandits)

class AsympUCB:
    def __init__(self, n_bandits):
        self.n_bandits = n_bandits

    def get_action(self,obs):
        mu = obs[::2]
        T = obs[1::2]
        T[T==0] = 1e-6
        log_f_t = np.log(1+T*np.log(T)**2)
        index = mu + np.sqrt(2*log_f_t/T)
        return np.argmax(index)
    
class MOSS:
    def __init__(self, n_bandits, horizon):
        self.n_bandits = n_bandits
        self.horizon = horizon

    def get_action(self,obs):
        mu = obs[::2]
        T = obs[1::2]
        T[T==0] = 1e-6
        log_plus = np.log(np.maximum(1,self.horizon/(self.n_bandits*T)))
        index = mu + np.sqrt(4*log_plus/T)
        return np.argmax(index)

class ExpertAsympUCB(AsympUCB):
    """
        Assign the index of all arms outside of expert_subgroup as min_index
    """
    def __init__(self, n_bandits, expert_subgroup, min_index = -1000):
        super().__init__(n_bandits)
        self.expert_subgroup = expert_subgroup
        self.min_index = min_index

    def get_action(self,obs):
        mu = obs[::2]
        T = obs[1::2]
        T[T==0] = 1e-6
        log_f_t = np.log(1+T*np.log(T)**2)
        index = mu + np.sqrt(2*log_f_t/T)
        mask = np.ones((self.n_bandits,))
        mask[self.expert_subgroup] = 0
        mask = mask.astype(bool)
        index[mask] = self.min_index
        return np.argmax(index)

class ExpertMOSS(MOSS):
    """
        Assign the index of all arms outside of expert_subgroup as min_index
    """
    def __init__(self, n_bandits, horizon, expert_subgroup, min_index = -1000):
        super().__init__(n_bandits, horizon)
        self.expert_subgroup = expert_subgroup
        self.min_index = min_index

    def get_action(self,obs):
        mu = obs[::2]
        T = obs[1::2]
        T[T==0] = 1e-6
        log_plus = np.log(np.maximum(1,self.horizon/(self.n_bandits*T)))
        index = mu + np.sqrt(4*log_plus/T)
        mask = np.ones((self.n_bandits,))
        mask[self.expert_subgroup] = 0
        mask = mask.astype(bool)
        index[mask] = self.min_index
        return np.argmax(index)

class Exp3:
    """
        reward in range [0,1]
        learning_rate: a function conditioned on n_bandits, horizon and time 't'
    """
    def __init__(self, n_bandits, horizon, learning_rate=None, is_reset=False, **kwargs):
        self.n_bandits = n_bandits
        self.horizon = horizon
        if learning_rate is None:
            self.learning_rate = self._default_learning_rate
        else:
            self.learning_rate = learning_rate
        self.reset()
        self.is_reset = is_reset
        if self.is_reset == False:
            self.n_switches = kwargs['n_switches']
            
    def reset(self):
        self.tracking_stats = np.zeros((self.n_bandits,)) #S_t
            
    def _default_learning_rate(self, n_bandits, horizon, t):
        return np.sqrt(2*np.log(n_bandits)/(n_bandits*horizon))

    def get_action(self, obs):
        if self.is_reset == True:
            horizon = self.horizon
        else:
            horizon = self.horizon*(self.n_switches+1)
        tmp = self.learning_rate(self.n_bandits, horizon, t=None)*self.tracking_stats
        # Max softmax trick
        tmp -= tmp.max()
        P_t = softmax(tmp)
        if np.isnan(np.sum(P_t)):
            import pdb; pdb.set_trace()
        return np.random.choice(self.n_bandits, p=P_t)
    
    def update(self, action, reward):
        P_t = softmax(self.learning_rate(self.n_bandits, self.horizon, t=None)*self.tracking_stats)
        self.tracking_stats += 1
        self.tracking_stats[action] -= (1-reward)/P_t[action]
    
class MetaAlg:
    """
        Proposal: "tracking" the best subgroup (store in tracking_stats).
        In each stochastic rolls-out, queries n_unbiased_obs samples from each arm.
        Then, uses the tracked statistic to select a subgroup.
        Next, runs the expert_alg (e.g: ExpertMOSS) on that subgroup.
        Finally, updates the tracking statistic with the n_bandits*n_unbiased_obs and the rolls-outed samples
        Repeat the process
        
        expert_subgroups: indices recommended by the experts. shape = (# of expert, subgroup's size)
    """
    def __init__(self, n_bandits, horizon, n_switches, n_unbiased_obs, alg_name, expert_subgroups, learning_rate=None, min_stats = -1000):
        self.n_bandits = n_bandits
        self.horizon = horizon
        self.n_unbiased_obs = n_unbiased_obs
        self.alg_name = alg_name
        self.expert_subgroups = expert_subgroups
        self.n_experts = self.expert_subgroups.shape[0]
        if learning_rate is None:
            self.learning_rate = self._default_learning_rate
        else:
            self.learning_rate = learning_rate
        self.tracking_stats = None #\hat{g}_n
        self.tracking_stats_counter = 0 # number of episodes till now
        self.collect_data_actions = np.repeat(np.arange(self.n_bandits),n_unbiased_obs)
        self.reset()
        self.min_stats = min_stats #must be smaller than min reward, for init tracking stats
        self.n_switches = n_switches

    def reset(self):
        self.collect_data_counter = 0
        self.cur_subgroup_index = -1
        self.cur_alg = None

    def _default_learning_rate(self, n_bandits, horizon, t):
        return np.sqrt(2*np.log(self.n_experts)/self.n_switches)

    def _update_tracking_stats(self, obs, selected_experts=None):
        """
        selected_experts: a vector to choose which expert to update
        """
        if selected_experts is None:
            selected_experts=np.arange(self.n_experts)
        # Update tracking stats with \hat{r}_n
        r_n = np.zeros((self.n_bandits,)) # mean reward vector
        for i in range(self.n_bandits):
            r_n[i] = obs[2*i]

        init_flag = False
        if self.tracking_stats is None:
            init_flag = True
            self.tracking_stats = np.zeros((self.n_experts,))
            self.tracking_stats_counter = 0
        count = self.tracking_stats_counter
        expert_list = np.intersect1d(selected_experts, np.arange(self.n_experts))
        for i in expert_list:
            subgroup_indices = self.expert_subgroups[i]
            g_ni = np.max(r_n[subgroup_indices])
            if init_flag == True:
                self.tracking_stats[i] = g_ni
            else:
                s_i = self.tracking_stats[i]*count + g_ni
                self.tracking_stats[i] = s_i/(count+1)
        
    def _select_subgroup(self):
        #EWA algorithm, max softmax trick
        tmp = self.learning_rate(self.n_bandits, self.horizon, t=None)*self.tracking_stats
        tmp -= tmp.max()
        P_t = softmax(tmp)
        self.cur_subgroup_index = np.random.choice(self.n_experts, p=P_t)
        cur_subgroup = self.expert_subgroups[self.cur_subgroup_index]
        if self.alg_name == 'ExpertAsympUCB':
            self.cur_alg = ExpertAsympUCB(self.n_bandits, cur_subgroup)
        elif self.alg_name == 'ExpertMOSS':
            self.cur_alg = ExpertMOSS(self.n_bandits, self.horizon, cur_subgroup)
        else:
            assert False, f"{self.alg_name} is NOT implemented."
    
    
    def get_action(self, obs): #get action for each rolls-out step
        if self.collect_data_counter == 0 and self.tracking_stats is not None: #Select subgroup
            self._select_subgroup()

        if self.collect_data_counter < self.n_bandits*self.n_unbiased_obs: #data collecting phase
            self.collect_data_counter += 1
            return self.collect_data_actions[self.collect_data_counter-1]
        
        if self.collect_data_counter == self.n_bandits*self.n_unbiased_obs: #Update non-selected subgroups
            self.collect_data_counter += 1
            if self.tracking_stats is None: #Init tracking_stats if needed
                self._update_tracking_stats(obs)
                self._select_subgroup()
            else:
                selected_experts = np.arange(self.n_experts)
                selected_experts = np.delete(selected_experts, self.cur_subgroup_index)
                self._update_tracking_stats(obs, selected_experts=selected_experts)
        return self.cur_alg.get_action(obs)
    
    def eps_end_update(self, obs): #update the tracking_stats after each rolls-out
        self._update_tracking_stats(obs, selected_experts=np.array([self.cur_subgroup_index]))
        self.tracking_stats_counter += 1
        self.reset()
