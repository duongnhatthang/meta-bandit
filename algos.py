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
    def __init__(self, n_bandits, horizon, is_reset=False, **kwargs):
        self.n_bandits = n_bandits
        self.horizon = horizon
        self.learning_rate = self._default_learning_rate()
        self.reset()
        self.is_reset = is_reset
        if self.is_reset == False:
            self.n_switches = kwargs['n_switches']
            
    def reset(self):
        self.tracking_stats = np.zeros((self.n_bandits,)) #S_t
            
    def _default_learning_rate(self):
        return np.sqrt(2*np.log(self.n_bandits)/(self.n_bandits*self.horizon))

    def get_action(self, obs):
        if self.is_reset == True:
            horizon = self.horizon
        else:
            horizon = self.horizon*(self.n_switches+1)
        # Max softmax trick
        tmp = self.learning_rate*self.tracking_stats
        tmp -= tmp.max()
        P_t = softmax(tmp)
        return np.random.choice(self.n_bandits, p=P_t)
    
    def update(self, action, reward):
        # Max softmax trick
        tmp = self.learning_rate*self.tracking_stats
        tmp -= tmp.max()
        P_t = softmax(tmp)
        self.tracking_stats += 1
        self.tracking_stats[action] -= (1-reward)/P_t[action]

class EWAmaxStats:
    """
        Proposal: "tracking" the best subgroup (store in tracking_stats).
        In each stochastic rolls-out, queries n_unbiased_obs samples from each arm.
        Then, uses the tracked statistic to select a subgroup.
        Next, runs the expert_alg (e.g: ExpertMOSS) on that subgroup.
        Finally, updates the tracking statistic with the value of max arm of each subgroup
        Repeat for all round
        
        expert_subgroups: indices recommended by the experts. shape = (# of expert, subgroup's size)
    """
    def __init__(self, n_bandits, horizon, n_switches, n_unbiased_obs, expert_subgroups, update_trick = False):
        self.n_bandits = n_bandits
        self.horizon = horizon
        self.n_unbiased_obs = n_unbiased_obs
        self.expert_subgroups = expert_subgroups
        self.n_experts = self.expert_subgroups.shape[0]
        self.n_switches = n_switches
        self.learning_rate = self._default_learning_rate()
        self.tracking_stats = np.zeros((self.n_experts,)) #\hat{g}_n
        self.collect_data_actions = np.repeat(np.arange(self.n_bandits),n_unbiased_obs)
        self.reset()
        self.update_trick = update_trick

    def reset(self):
        self.collect_data_counter = 0
        self.cur_subgroup_index = -1
        self.cur_alg = None

    def _default_learning_rate(self):
        return np.sqrt(2*np.log(self.n_experts)/(self.n_switches+1))

    def _update_tracking_stats(self, obs, selected_experts=None):
        """
        selected_experts: a vector to choose which expert to update
        """
        if selected_experts is None:
            selected_experts=np.arange(self.n_experts)
        # Update tracking stats with \hat{r}_n
        r_n = obs[::2] # mean reward vector
        expert_list = np.intersect1d(selected_experts, np.arange(self.n_experts))
        for i in expert_list:
            subgroup_indices = self.expert_subgroups[i]
            g_ni = np.max(r_n[subgroup_indices])
            self.tracking_stats[i] += g_ni

    def _select_subgroup(self):
        #EWA algorithm, max softmax trick
        tmp = self.learning_rate*self.tracking_stats
        tmp -= tmp.max()
        P_t = softmax(tmp)
        self.cur_subgroup_index = np.random.choice(self.n_experts, p=P_t)
        cur_subgroup = self.expert_subgroups[self.cur_subgroup_index]
        self.cur_alg = ExpertMOSS(self.n_bandits, self.horizon, cur_subgroup)
    
    def get_action(self, obs): #get action for each rolls-out step
        if self.collect_data_counter == 0 and (self.tracking_stats == np.zeros((self.n_experts,))).all() == False: #Select subgroup
            self._select_subgroup()

        if self.collect_data_counter < self.n_bandits*self.n_unbiased_obs: #data collecting phase
            self.collect_data_counter += 1
            return self.collect_data_actions[self.collect_data_counter-1]
        
        if self.collect_data_counter == self.n_bandits*self.n_unbiased_obs: #Update non-selected subgroups
            self.collect_data_counter += 1 #Only be here once per switch
            if (self.tracking_stats == np.zeros((self.n_experts,))).all(): #Init tracking_stats if needed
                self._update_tracking_stats(obs)
                self._select_subgroup()
            elif self.update_trick == False:
                selected_experts = np.arange(self.n_experts)
                selected_experts = np.delete(selected_experts, self.cur_subgroup_index)
                self._update_tracking_stats(obs, selected_experts=selected_experts)
        return self.cur_alg.get_action(obs)
    
    def eps_end_update(self, obs): #update the tracking_stats after each rolls-out
        if self.update_trick == False:
            self._update_tracking_stats(obs, selected_experts=np.array([self.cur_subgroup_index]))
        else:
            self._update_tracking_stats(obs)
        self.reset()

class PhaseElim:
    def __init__(self, n_bandits, horizon, C=1, min_index = -1000):
        self.n_bandits = n_bandits
        self.horizon = horizon
        self.C = C
        self.reset()
        self.min_index = min_index
    
    def reset(self):
        self.ml_counter = -1
        self.cur_l = 1 #phase counter
        self.A_l = np.arange(self.n_bandits)
        self.cur_mu = np.zeros((self.n_bandits,)) # Tracking mu in one phase
        self.cur_phase_actions = np.repeat(self.A_l,self._get_ml()) #get 'ml' observations from each arm in the self.A_l

    def _get_ml(self):
        return round(self.C*2**(2*self.cur_l)*np.log(max(np.exp(1),self.n_bandits*self.horizon*2**(-2*self.cur_l))))
    
    def _eliminate(self):
        if self.A_l.shape[0]==1:
            return
        max_mu = np.max(self.cur_mu)
        eliminate_arm_index = np.where(self.cur_mu + 2**(-self.cur_l) < max_mu)[0]
        self.A_l = np.setdiff1d(self.A_l,eliminate_arm_index) # yields the elements in `self.A_l` that are NOT in `eliminate_arm_index`
        self.cur_mu = np.zeros((self.n_bandits,))
        self.cur_mu[eliminate_arm_index] = self.min_index

    def get_action(self,obs):
        if self.ml_counter == self._get_ml()*self.A_l.shape[0]-1:
            # Reset statistics when starting a new phase
            self.ml_counter = -1
            self.cur_l += 1
            self._eliminate()
            self.cur_phase_actions = np.repeat(self.A_l,self._get_ml())
        self.ml_counter += 1
        return self.cur_phase_actions[self.ml_counter]

    def update(self, action, reward):
        self.cur_mu[action] += reward/self._get_ml()
        
class MetaPElargeGap: #Phase Elimination with big Gap (always only return 1 arm after PE in each round)
    def __init__(self, n_bandits, horizon, n_switches, expert_subgroups, C=1, min_index = -1000):
        self.min_index = min_index
        self.n_bandits = n_bandits
        self.horizon = horizon
        self.expert_subgroups = expert_subgroups
        self.n_experts = self.expert_subgroups.shape[0]
        self.n_switches = n_switches
        self.PE_algo = PhaseElim(n_bandits, horizon, C, min_index)
        self.reset()
        self.MOSS_algo = MOSS(self.n_bandits, self.horizon)
        self.subgroup_size = expert_subgroups[0].shape[0]
        self.C1 = np.sqrt(horizon*self.subgroup_size)
        self.C2 = np.sqrt(horizon*n_bandits)
        self.C3 = horizon
        self._set_is_explore()

    def reset(self):
        self.PE_algo.reset()
        self.predicted_opt_arms = []
        self.cur_switch = 0

    def _get_delta_n(self):
        if self.n_switches+1 == self.cur_switch:
            return 1
        numerator = self.C3 - self.C1
        denominator = (self.C2 - self.C1)*2*(self.n_switches+1-self.cur_switch-1)
        return np.sqrt(numerator/max(1e-6,denominator))

    def _set_is_explore(self):
        p = self._get_delta_n()
        p = min(p,1) # TODO: prob bug here
        self.is_explore = bool(np.random.choice(2,p=[1-p,p]))

    def get_action(self, obs): #get action for each rolls-out step
        # The 2nd condition to make sure it still run with small Gap (PE return multiple arms)
        if self.is_explore == True and len(self.predicted_opt_arms)<self.subgroup_size:
            return self.PE_algo.get_action(obs)
        else:
            return self.MOSS_algo.get_action(obs)
    
    def eps_end_update(self, obs): #update the tracking_stats after each rolls-out
        # The 2nd condition to make sure it still run with small Gap (PE return multiple arms)
        if self.is_explore == True and len(self.predicted_opt_arms)<self.subgroup_size:
            arms_found = self.PE_algo.A_l
            if arms_found.shape[0]==1:
                if arms_found[0] not in self.predicted_opt_arms:
                    self.predicted_opt_arms.append(arms_found[0])
            else:
#                 print(f"{arms_found.shape[0]} arms left after elimination: {self.PE_algo.cur_mu}.")
                self.predicted_opt_arms += arms_found.tolist()
                self.predicted_opt_arms = list(set(self.predicted_opt_arms))
            self.MOSS_algo = ExpertMOSS(self.n_bandits, self.horizon, self.predicted_opt_arms, self.min_index)
        self.PE_algo.reset()
        self.cur_switch += 1
        self._set_is_explore()

    def update(self, action, reward):
        # The 2nd condition to make sure it still run with small Gap (PE return multiple arms)
        if self.is_explore == True and len(self.predicted_opt_arms)<self.subgroup_size:
            self.PE_algo.update(action, reward)

class MetaPM: # PE + Partial Monitoring
    """
        Tracking the statistic of each EXT experts and 1 EXR expert => EWA.
         - If EXT expert contain PE survival arms => f(C1) cost, else => f(C3) cost
         - EXR expert => f(C2) cost
         - Only update statistic at EXR round
    """
    def __init__(self, n_bandits, horizon, n_switches, expert_subgroups, C=1, min_index = -1000, is_small_C1=True):
        self.n_bandits = n_bandits
        self.horizon = horizon
        self.n_switches = n_switches
        self.n_round = self.n_switches+1
        self.expert_subgroups = expert_subgroups
        self.subgroup_size = expert_subgroups[0].shape[0]
        self.n_experts = self.expert_subgroups.shape[0]
        self.min_index = min_index
        self.C1 = np.sqrt(horizon*self.subgroup_size)
        self.C2 = np.sqrt(horizon*n_bandits)
        self.C3 = horizon
        self.learning_rate = self._default_learning_rate()
        self.tracking_stats = np.zeros((self.n_experts+1,)) # Last expert is EXR
        self.tracking_stats[-1] = 1 #TODO: test prior forcing exploration
        self.PE_algo = PhaseElim(n_bandits, horizon, C, min_index)
        self.reset()
        self.delta_n = self._get_delta_n()
        assert self.delta_n <= 1 and self.delta_n >= 0, f" self.delta_n ({self.delta_n}) is not in the range [0,1]. Reduce N_EXPERT, HORIZON or increase N_SWITCHES."
        self.is_small_C1 = is_small_C1
        self._select_expert()
        print(f'MetaPM: self.delta_n = {self.delta_n}, self.learning_rate={self.learning_rate}')
        assert self.C1 <= self.C2 and self.C2 <= self.C3, f"C1 ({self.C1}) < C2 ({self.C2}) < C3 ({self.C3}) not satisfied."

    def reset(self):
        self.PE_algo.reset()

    def _default_learning_rate(self):
        return 1

    def _get_delta_n(self):
#         return ((np.log(self.n_experts)*self.C3**2)/(self.n_round*self.C2**2))**(1/3)
        return (self.C3*np.log(self.n_experts)/(self.C2*self.n_round))**(1/2)

    def _select_expert(self):
        #EWA algorithm, max softmax trick
        tmp = self.learning_rate*self.tracking_stats
        tmp -= tmp.max()
        Q_n = softmax(tmp)
        P_n = np.zeros((self.n_experts+1,))
        P_n[-1] = self.delta_n
        self.P_n = P_n + (1-self.delta_n)*Q_n # Expert distribution to select/sample from
        if (self.P_n==0).any(): # fix 0 probability
            self.P_n[self.P_n==0]=1e-6
            self.P_n/=np.sum(self.P_n)
        self.cur_subgroup_index = np.random.choice(self.n_experts+1, p=self.P_n)
        if self.cur_subgroup_index < self.n_experts: # EXT: exploit
            cur_subgroup = self.expert_subgroups[self.cur_subgroup_index]
            self.cur_algo = ExpertMOSS(self.n_bandits, self.horizon, cur_subgroup)
        else: # EXR: explore
            self.cur_algo = self.PE_algo

    def get_action(self, obs): #get action for each rolls-out step
        return self.cur_algo.get_action(obs)

    def eps_end_update(self, obs): #update the tracking_stats after each rolls-out
        self._update_tracking_stats(obs)
        self._select_expert()
        self.reset()

    def _get_tilda_c_n(self):
        if self.cur_subgroup_index < self.n_experts: # EXT: exploit
            return None
        # EXR: exploration case
        tilda_c_n = np.zeros((self.n_experts+1,))
        tilda_c_n[-1] = self.C2
        surviving_arms = self.PE_algo.A_l
        experts_contains_surviving_arms = []
        for i, e in enumerate(self.expert_subgroups):
            tmp = np.intersect1d(surviving_arms, e)
            if len(tmp)>0:
                experts_contains_surviving_arms.append(i)
        tilda_c_n[np.arange(self.n_experts).astype(int)] = self.C3
        tilda_c_n[experts_contains_surviving_arms] = self.C1
        if self.is_small_C1 == True:
            tilda_c_n -= self.C1
            tilda_c_n /= self.P_n[-1]
        return tilda_c_n

    def _get_loss_vector(self):
        tilda_c_n = self._get_tilda_c_n()
        if tilda_c_n is None:
            return None
        l_n = self.delta_n*tilda_c_n/self.C3
        return l_n

    def _update_tracking_stats(self, obs):
        l_n = self._get_loss_vector()
        if l_n is not None:
            self.tracking_stats += 1-l_n

    def update(self, action, reward):
        if self.cur_subgroup_index == self.n_experts: # EXR: exploration
            self.PE_algo.update(action, reward)

class MetaPMtrick(MetaPM):
    """
    Trick:  - Remove (stop tracking) all experts not contain the surviving arms returned by Phase Elimination
    """
    def __init__(self, n_bandits, horizon, n_switches, expert_subgroups, C=1, min_index = -1000, is_small_C1=False):
        super().__init__(n_bandits, horizon, n_switches, expert_subgroups, C, min_index, is_small_C1)
        self.surviving_experts = np.arange(self.n_experts)

    def _update_tracking_stats(self, obs):
        l_n = self._get_loss_vector()
        if l_n is not None:
            if self.is_small_C1 == True:
                surviving_experts = np.where(l_n==0)[0]
            else:
                surviving_experts = np.where(l_n==self.delta_n*self.C1/self.C3)[0]
            self.surviving_experts = np.intersect1d(self.surviving_experts, surviving_experts)
            if self.surviving_experts.shape[0]==1: #stop Exploration after finding the correct expert
                cur_subgroup = self.expert_subgroups[self.surviving_experts[0]]
                self.cur_algo = ExpertMOSS(self.n_bandits, self.horizon, cur_subgroup)
            else:
                temp = np.ones_like(self.tracking_stats)*self.min_index
                temp[self.surviving_experts] = self.tracking_stats[self.surviving_experts]
                temp[-1] = self.tracking_stats[-1] #EXR expert statistic
                self.tracking_stats = temp
                self.tracking_stats += 1-l_n
#         print(f'MetaPMtrick: self.surviving_experts = {self.surviving_experts}')
        
    def eps_end_update(self, obs): #update the tracking_stats after each rolls-out
        self._update_tracking_stats(obs)
        if self.surviving_experts.shape[0]>1: #Only EWA to select expert if there are more than 1 surviving
            self._select_expert()
        self.reset()