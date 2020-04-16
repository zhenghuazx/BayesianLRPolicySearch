import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, Lambda, Input, Subtract, Activation
from keras.models import Model, Sequential
from keras import backend as K
from keras import utils as np_utils
from keras import optimizers
from simulation_model import chromatography, fermentation_simulator
import matplotlib.pyplot as plt
import pymc3 as pm
from scipy.stats import beta

class Agent(object):

    def __init__(self, input_dim, output_dim, hidden_dims=[16, 32]):
        """Gym Playing Agent
        Args:
            input_dim (int): the dimension of state.
                Same as `env.observation_space.shape[0]`
            output_dim (int): the number of discrete actions
                Same as `env.action_space.n`
            hidden_dims (list): hidden dimensions
        Methods:
            private:
                __build_train_fn -> None
                    It creates a train function
                    It's similar to defining `train_op` in Tensorflow
                __build_network -> None
                    It create a base model
                    Its output is each action probability
            public:
                get_action(state) -> action
                fit(state, action, reward) -> None
        """

        self.state_size = input_dim
        self.action_size = output_dim
        self.discount_factor = 0.99
        self.learning_rate = 0.01
        self.rooling_window_length = 500
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.actions = []
        self.posteriors = []
        self.models = []
        self.model = self.build_model()
        self.__build_train_fn()
        self.normalizer = 30
        self.normalized_states = []

    def build_model(self):
        model = Sequential()
        model.add(Dense(16, input_dim=self.state_size, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(self.action_size, activation='softmax'))
        # model.add(Dense(self.action_size,input_dim=self.state_size, activation='softmax'))
        model.summary()
        return model

    def __build_train_fn(self):
        """Create a train function
        It replaces `model.fit(X, y)` because we use the output of model and use it for training.
        For example, we need action placeholder
        called `action_one_hot` that stores, which action we took at state `s`.
        Hence, we can update the same action.
        This function will create
        `self.train_fn([state, action_one_hot, discount_reward])`
        which would train the model.
        """
        action_onehot_placeholder = K.placeholder(shape=(None,self.action_size),
                                                  name="action_onehot")
        discount_reward_placeholder = K.placeholder(shape=(None,),
                                                    name="discount_reward")

        #likelihood_ratio_placeholder = K.placeholder(shape=(None,),
        #                                            name="likelihood_ratio")
        action_prob = K.sum(action_onehot_placeholder * self.model.output, axis=1)
        cross_entropy = K.log(action_prob) * discount_reward_placeholder
        loss = -K.mean(cross_entropy)

        # action_prob = K.sum(self.model.output * action_onehot_placeholder, axis=1)
        # log_action_prob = K.log(action_prob)
        #
        #
        # loss = - log_action_prob * discount_reward_placeholder# * likelihood_ratio_placeholder
        # loss = K.mean(loss)

        adam = optimizers.Adam(lr=self.learning_rate)

        updates = adam.get_updates(params=self.model.trainable_weights,
                                   #constrait=[self.model.output],
                                   loss=loss)

        self.train_fn = K.function(inputs=[self.model.input,
                                           action_onehot_placeholder,
                                           #likelihood_ratio_placeholder,
                                           discount_reward_placeholder],
                                   outputs=[],
                                   updates=updates)

    def memorize(self, state, action, reward, posterior, model_weight, normalized_states):
        self.actions.append(action)
        if posterior is not None:
            self.posteriors.append(posterior)
            self.models.append(model_weight)

        self.states.append(state)
        self.normalized_states.append(normalized_states)
        self.rewards.append(reward)

    def get_action(self, state):
        """Returns an action at given `state`
        Args:
            state (1-D or 2-D Array): It can be either 1-D array of shape (state_dimension, )
                or 2-D array shape of (n_samples, state_dimension)
        Returns:
            action: an integer action value ranging from 0 to (n_actions - 1)
        """
        shape = state.shape

        if len(shape) == 1:
            assert shape == (self.state_size,), "{} != {}".format(shape, self.state_size)
            state = np.expand_dims(state, axis=0)

        elif len(shape) == 2:
            assert shape[1] == (self.state_size), "{} != {}".format(shape, self.state_size)

        else:
            raise TypeError("Wrong state shape is given: {}".format(state.shape))

        action_prob = np.squeeze(self.model.predict(state))
        assert len(action_prob) == self.action_size, "{} != {}".format(len(action_prob), self.action_size)
        return np.random.choice(np.arange(self.action_size), p=action_prob), action_prob

    def fit(self, S, A, R, chroma, posterior, action_prob):
        """Train a network
        Args:
            S (2-D Array): `state` array of shape (n_samples, state_dimension)
            A (1-D Array): `action` array of shape (n_samples,)
                It's simply a list of int that stores which actions the agent chose
            R (1-D Array): `reward` array of shape (n_samples,)
                A reward is given after each action.
        """
        discount_reward = compute_discounted_R(R)

        likelihoods = likelihood(chroma.posterior, posterior, S, A)
        trajectory_dist = [action_prob[i] * likelihoods[i] for i in range(len(likelihoods))] #  np.product(np.append(action_prob,likelihoods))
        trajectory_dist = compute_forward_ratio(trajectory_dist)
        mixture_trajectory_dist = 0
        for hist_idx in range(len(self.actions)):
            action_probs = self.models[hist_idx].predict(S)
            mixture_action_prob = [action_probs[step, np.argmax(A[step,])] for step in range(len(A))]
            mixture_likelihoods = likelihood(chroma.posterior, self.posteriors[int(hist_idx / 100)], S, A) # [likelihood(p, S, A) for p in self.posteriors]
            mixture_trajectory_dist = [1/len(self.posteriors) * mixture_action_prob[i] * mixture_likelihoods[i] for i in range(len(mixture_likelihoods))]
            # 1/len(self.posteriors) * np.prod(np.append(mixture_action_prob, mixture_likelihoods))
            mixture_trajectory_dist = compute_forward_ratio(mixture_trajectory_dist)


        assert S.shape[1] == self.state_size, "{} != {}".format(S.shape[1], self.state_size)
        assert len(discount_reward.shape) == 1, "{} != 1".format(len(discount_reward.shape))
        BLR = [trajectory_dist[i] / (mixture_trajectory_dist[i] + 1e-7) for i in range(len(trajectory_dist))]
        #print(BLR)

        weighted_discount_reward = np.multiply(BLR, discount_reward)
        W = np.asarray(weighted_discount_reward)
        self.train_fn([S[:-1, ], A, W])

    def fit2(self, S, A, R, chroma, posterior, normalized_states):
        """Train a network
        Args:
            S (2-D Array): `state` array of shape (n_samples, state_dimension)
            A (1-D Array): `action` array of shape (n_samples,)
                It's simply a list of int that stores which actions the agent chose
            R (1-D Array): `reward` array of shape (n_samples,)
                A reward is given after each action.
        """

        action_probs = [0] * len(A)
        discount_reward = [0] * len(A)
        likelihoods = [0] * len(A)
        trajectory_dists = [0] * len(A)
        mixture_trajectory_dists = [0] * len(A)
        weighted_discount_reward = [0] * len(A)
        for i in range(len(A)):
            action_prob = self.model.predict(normalized_states[i])
            action_probs[i] = action_prob
            discount_reward[i] = compute_discounted_R(R[i])
            likelihoods[i] = likelihood(chroma.posterior, posterior, S[i], A[i])
            trajectory_dist = [action_prob[k][np.argmax(A[i][k,])] * likelihoods[i][k] for k in range(len(likelihoods[i]))]  # np.product(np.append(action_prob,likelihoods))
            trajectory_dist = compute_forward_ratio(trajectory_dist)
            trajectory_dists[i] = trajectory_dist

            mixture_trajectory_dist_hist = []
            for hist_idx in range(max(len(self.posteriors)-int(self.rooling_window_length/len(A)), 0), len(self.posteriors)): # range(len(self.actions) - self.rooling_window_length, len(self.actions)):
                #print(hist_idx)
                #print(len(self.models))
                mixture_action_probs = self.models[hist_idx].predict(normalized_states[i])
                mixture_action_prob = [mixture_action_probs[step, np.argmax(A[i][step,])] for step in range(len(A[i]))]
                mixture_likelihoods = likelihood(chroma.posterior, self.posteriors[hist_idx], S[i], A[i])  # [likelihood(p, S, A) for p in self.posteriors]
                mixture_trajectory_dist = [1 / len(self.posteriors) * mixture_action_prob[k] * mixture_likelihoods[k] for k
                                           in range(len(mixture_likelihoods))]
                mixture_trajectory_dist_hist.append(compute_forward_ratio(mixture_trajectory_dist))

            mixture_trajectory_dist_hist = np.asarray(mixture_trajectory_dist_hist)

            BLR = [trajectory_dist[k] / (np.sum(mixture_trajectory_dist_hist[:,k])) for k in range(len(trajectory_dist))]


            weighted_discount_reward[i] = discount_reward[i] # np.multiply(BLR, discount_reward[i]) # discount_reward[i]
            # S[i] = S[i][:-1, ]
            normalized_states[i] = normalized_states[i][:-1,]

        S = np.reshape(np.asarray(normalized_states), (np.asarray(normalized_states).shape[0]*np.asarray(normalized_states).shape[1], np.asarray(normalized_states).shape[2]))
        A = np.reshape(np.asarray(A), (np.asarray(A).shape[0]*np.asarray(A).shape[1], np.asarray(A).shape[2]))
        W = np.asarray(weighted_discount_reward)
        # print(W)
        W = np.reshape(W, (W.shape[0]*W.shape[1]))

        self.train_fn([S, A, W])

def likelihood(probability, posterior_sample_idx, state, action):
    H = state.shape[0]
    likelihoods = []
    for i in range(H-1):
        likelihoods.append(single_step_likelihood(probability, posterior_sample_idx, state[i+1,], state[i,], action[i,],i))
    return likelihoods

def single_step_likelihood(probability, posterior_sample_idx, next_state, state, action, t):
    if np.isscalar(action) is not True:
        action = np.argmax(action)
    prob_protein = probability[t][0][action][:,posterior_sample_idx]
    prob_impurity = probability[t][1][action][:,posterior_sample_idx]
    protein_likelihood = beta.pdf(next_state[0]/state[0], a=prob_protein[0], b=prob_protein[1])
    impurity_likelihood = beta.pdf(next_state[1]/state[1], a=prob_impurity[0], b=prob_impurity[1])
    return protein_likelihood * impurity_likelihood

def compute_discounted_R(R, discount_rate=1):
    """Returns discounted rewards
    Args:
        R (1-D array): a list of `reward` at each time step
        discount_rate (float): Will discount the future value by this rate
    Returns:
        discounted_r (1-D array): same shape as input `R`
            but the values are discounted
    Examples:
        #>>> R = [1, 1, 1]
        #>>> compute_discounted_R(R, .99) # before normalization
        [1 + 0.99 + 0.99**2, 1 + 0.99, 1]
    """
    discounted_r = np.zeros_like(R, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(R))):

        running_add = running_add * discount_rate + R[t]
        discounted_r[t] = running_add
    discounted_r += 68
    discounted_r /= 88

    return discounted_r

def compute_forward_ratio(L):
    discounted_r = np.zeros_like(L, dtype=np.float32)
    running_prod = 1
    for t in range(len(L)):
        running_prod = running_prod * L[t]
        discounted_r[t] = running_prod

    #discounted_r -= (discounted_r.mean() / discounted_r.std())
    return discounted_r


def reward_function(protein, impurity, required_ratio=0.85, required_protein=8):
      Cf= 48
      Cl = 6
      price_prod = 5
      if protein/(protein+max(impurity,0)) < required_ratio:
        return -Cf
      elif protein < required_protein:
        return price_prod * protein - Cl*(required_protein - protein)
      else:
        return price_prod * required_protein



def DP_Policy(s):
    protein_max = 30
    impurity_max = 30
    step_size = 0.02
    protein = np.arange(0.01, 1, step_size) * protein_max
    impurity = np.arange(0.01, 1, step_size) * impurity_max
    protein_idx = int(s[0] / protein_max / step_size)
    impurity_indx = int(s[1] / protein_max / step_size)
    return policy[int(s[2]), protein_idx, impurity_indx]

def run_episode_dp(chroma, agent,seed, save_model, nj=50):
    normalizer = 30
    # set up environment
    time_span = 10.
    t = np.linspace(0., 50., 501)
    t_realization = np.linspace(0., 50., 51) * time_span
    sample_idx = [int(i) for i in t_realization]
    total_reward = 0
    total_reward_dp = 0
    for episode in range(nj):
        # simulate from environment
        np.random.seed(int(str(seed) + str(episode)))
        alpha1, alpha2 = np.random.normal(0.11, 0.01), np.random.normal(0.11, 0.01)
        Feed = 30 + np.random.normal(0, 5)
        Si = 780 + np.random.normal(0, 40)
        S = 40 + np.random.normal(0, 2)
        action = [Feed] * len(sample_idx)
        if episode % 100 == 0:
            chroma.build_posterior()
            chroma.postreior_sampler()
        fs = fermentation_simulator(action,time_span=time_span, N=100)
        fs.s0 = [0.5,S]
        fs.Si = Si
        simulation_out = fs.simulate(t, Feed, sample_idx)[1]
        upstream_out = simulation_out[-1,0] + np.random.normal(0,simulation_out[-1,0] / 256)
        initial_state = [upstream_out * alpha1, upstream_out * alpha2]
        print(initial_state)

        S = []
        A = []
        R = []
        s = np.append(initial_state, 0)
        S.append(s)
        for h in range(chroma.horizon):
            # on-policy
            a = DP_Policy(s)
            print(a)
            s = np.squeeze(chroma.simulate(a, s, h + 1))
            s = np.append(s, h + 1)

            if h == chroma.horizon - 1:
                r = -10 + reward_function(s[0], s[1])
            else:
                r = -10
            total_reward += r

    return total_reward / nj

def run_episode(chroma, agent,seed, save_model, nj=50):
    normalizer = 30
    # set up environment
    time_span = 10.
    t = np.linspace(0., 50., 501)
    t_realization = np.linspace(0., 50., 51) * time_span
    sample_idx = [int(i) for i in t_realization]
    total_reward = 0
    total_reward_dp = 0
    for episode in range(nj):
        # simulate from environment
        np.random.seed(int(str(seed) + str(episode)))
        alpha1, alpha2 = np.random.normal(0.11, 0.01), np.random.normal(0.11, 0.01)
        Feed = 30 + np.random.normal(0, 5)
        Si = 780 + np.random.normal(0, 40)
        S = 40 + np.random.normal(0, 2)
        action = [Feed] * len(sample_idx)

        fs = fermentation_simulator(action,time_span=time_span, N=100)
        fs.s0 = [0.5,S]
        fs.Si = Si
        simulation_out = fs.simulate(t, Feed, sample_idx)[1]
        upstream_out = simulation_out[-1,0] + np.random.normal(0,simulation_out[-1,0] / 256)
        initial_state = [upstream_out * alpha1, upstream_out * alpha2]
        print(initial_state)

        S = []
        A = []
        R = []
        A_prob = []
        s = np.append(initial_state, 0)
        S.append(s)
        normalized_states = [np.array([s[0] / normalizer, s[1] / normalizer, s[2]])]

        for h in range(chroma.horizon):
            # on-policy
            action, action_prob = agent.get_action(np.array([s[0]/normalizer, s[1]/normalizer, s[2]]))
            a = np.zeros([agent.action_size])
            print(action)
            a[action] = 1
            s = np.squeeze(chroma.simulate(action, s, h, ))
            s = np.append(s, h + 1)
            normalized_s = np.array([s[0]/normalizer, s[1]/normalizer, s[2]])
            if h == chroma.horizon - 1:
                r = -10 + reward_function(s[0], s[1])
            else:
                r = -10

            total_reward += r
            S.append(s)
            A.append(a)
            R.append(r)
            normalized_states.append(normalized_s)
            A_prob.append(action_prob[action])
            if h == chroma.horizon - 1:
                S = np.array(S)
                A = np.array(A)
                R = np.array(R)
                normalized_states = np.array(normalized_states)
                # print(normalized_states)
                if episode == nj-1: # save_model:
                    agent.memorize(S, A, R, chroma.posterior_sample_idx, agent.model, normalized_states)
                else:
                    agent.memorize(S, A, R, None, None, normalized_states)
                # first 100 episodes do not reuse previous samples
                # if len(agent.actions) < 100:
                # #     # off-policy
                # #     S = agent.states[-agent.rooling_window_length:]
                # #     A = agent.actions[-agent.rooling_window_length:]
                # #     R = agent.rewards[-agent.rooling_window_length:]
                # #     agent.fit2(S, A, R, chroma.posterior)
                # # else:
                #     agent.fit(S, A, R, chroma, chroma.posterior_sample_idx, A_prob)
        if episode == nj - 1:
                # off-policy
                S_backward = agent.states[-agent.rooling_window_length:]
                A_backward = agent.actions[-agent.rooling_window_length:]
                R_backward = agent.rewards[-agent.rooling_window_length:]
                normalized_states_backward = agent.normalized_states[-agent.rooling_window_length:]
                agent.fit2(S_backward, A_backward, R_backward, chroma, chroma.posterior_sample_idx, normalized_states_backward)

    return total_reward / nj

def transition_prob(chroma, t, i, j, a, i_next, j_next, protein,impurity, protein_max, impurity_max, step_size):
    protein_interval = [protein[i_next], min(protein[i_next] + protein_max * step_size, protein_max)]
    impurity_interval = [impurity[j_next], min(impurity[j_next] + impurity_max * step_size, impurity_max)]

    protein_possible = [protein[i] * chroma.posterior[t+1]['protein'][a][0], protein[i] * chroma.posterior[t+1]['protein'][a][1]]
    impurity_possible = [impurity[i] * chroma.posterior[t+1]['impurity'][a][0], impurity[i] * chroma.posterior[t+1]['impurity'][a][1]]

    protein_interval = pd.Interval(protein_interval[0], protein_interval[1],closed='both')
    impurity_interval = pd.Interval(impurity_interval[0], impurity_interval[1],closed='both')
    protein_possible = pd.Interval(protein_possible[0], protein_possible[1],closed='both')
    impurity_possible = pd.Interval(impurity_possible[0], impurity_possible[1],closed='both')

    protein_prob = (min(protein_interval.right,protein_possible.right) - max(protein_possible.left, protein_interval.left))/protein[i] if protein_interval.overlaps(protein_possible) else 0
    protein_prob /= chroma.posterior[t + 1]['protein'][a][1] - chroma.posterior[t + 1]['protein'][a][0]
    impurity_prob = (min(impurity_interval.right, impurity_possible.right) - max(impurity_possible.left, impurity_interval.left))/impurity[i] if impurity_interval.overlaps(impurity_possible) else 0
    impurity_prob /= chroma.posterior[t+1]['impurity'][a][1]- chroma.posterior[t+1]['impurity'][a][0]
    # protein_prob = (protein_interval[1] - protein_interval[0]) / (chroma.posterior[t+1]['protein'][a][1] - chroma.posterior[t+1]['protein'][a][0])
    #impurity_prob = (impurity_interval[1] - impurity_interval[0]) / (chroma.posterior[t+1]['impurity'][a][1]- chroma.posterior[t+1]['impurity'][a][0])
    #print(protein_prob)
    #print(impurity_prob)
    return protein_prob * impurity_prob

def look_ahead_total_reward(chroma, t, i, j, action, V, protein, impurity, protein_max, impurity_max, step_size):
    expected_total_reward = np.zeros(len(action))
    for i_next in range(len(protein)):
        for j_next in range(len(impurity)):
            for a in action:
                expected_total_reward[a] = expected_total_reward[a] + transition_prob(chroma, t, i, j, a, i_next, j_next, protein, impurity, protein_max, impurity_max, step_size) * V[t + 1, i_next, j_next]
    return expected_total_reward

def DP(chroma,H, impurity, protein, action, protein_max, impurity_max, step_size):
    V = np.zeros(shape=(H, len(protein), len(impurity)))
    policy = np.zeros(shape=(H, len(protein), len(impurity)))
    for t in reversed(range(H)):
        for i in range(len(protein)):
            for j in range(len(impurity)):
                print('t:{},i:{},j:{}'.format(t,i,j))
                if t == H-1:
                    V[t, i, j] = reward_function(protein[i], impurity[j])
                else:
                    expected_reward = -10 + look_ahead_total_reward(chroma, t, i, j, action, V, protein, impurity, protein_max, impurity_max, step_size)
                    optimal_action = np.argmax(expected_reward)
                    termination_reward = reward_function(protein[i], impurity[j])
                    continue_best_reward = expected_reward[optimal_action]
                    V[t, i, j] = continue_best_reward
                    policy[t, i, j] = optimal_action
                    if termination_reward < continue_best_reward:
                        V[t, i, j] = continue_best_reward
                        policy[t, i, j] = optimal_action
                    else:
                        V[t, i, j] = termination_reward
                        policy[t, i, j] = -1

    return V, policy

#
import matplotlib.pyplot as plt


fig, ax = plt.subplots()
im = ax.imshow(np.transpose(V[2,:,:]), origin='lower')
ax.set_title("Harvest of local farmers (in tons/year)")
fig.tight_layout()
plt.show()


protein_max = 30
impurity_max = 30
step_size = 0.02
protein = np.arange(0.01,1,step_size) * protein_max
impurity = np.arange(0.01,1,step_size) * impurity_max
H = 3
required_ratio = 0.85
required_protein = 8
T1 = 11
action_down = list(range(10))

V, policy = DP(chroma, H, impurity, protein, action_down, protein_max, impurity_max, step_size)


def main(chroma):
    agent = Agent(3, 10, [16])
    # time_span = 10.
    # t = np.linspace(0., 50., 501)
    # t_realization = np.linspace(0., 50., 51) * time_span
    # sample_idx = [int(i) for i in t_realization]
    # std = 10.
    # alpha1, alpha2 = np.random.normal(0.1,0.01), np.random.normal(0.1,0.01)
    reward_result = []
    for batch in range(260):
        # feed = np.random.uniform(20,40)
        # action = [feed] * len(sample_idx)
        # # if batch % 100 == 0:
        # #     chroma.build_posterior()
        # #     chroma.postreior_sampler()
        # fs = fermentation_simulator(action, time_span, N=100)
        # simulation_out = fs.simulate(t, feed, sample_idx)[1]
        # initial_state = [simulation_out[-1, 0] * alpha1, simulation_out[-1, 0] * alpha2]
        reward = run_episode(chroma, agent, batch, batch % 100 == 0)
        chroma.update_posterior_sample()
        reward_result.append(reward)
        print("iteration: {}, reward: {:0.2f}".format(batch, reward))

    return reward_result

   # -54.83871682909874
if __name__ == '__main__':
    chroma = chromatography()
    chroma.build_posterior()
    reward_result_PG = main(chroma)
    reward_result_PG = main(chroma)
    plt.plot(range(len(reward_result[:250])), reward_result[:250], 'b-', lw=2, label='BRAMPS')
    plt.plot(range(len(reward_result_PG[:250])), reward_result_PG[:250], 'r-', lw=2, label='PG')
    plt.ylim(ymax=40)
    plt.xlabel('Iteration', fontsize=15)
    plt.ylabel('Average Reward', fontsize=15)
    plt.legend(fontsize=15)
    plt.axvline(x=150, linestyle='dashed', lw=2)
    plt.show()
    import scipy
    scipy.stats.ttest_ind(reward_result[150:250], reward_result_PG[150:250], equal_var=False)