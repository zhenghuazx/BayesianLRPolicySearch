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
import scipy
from random import sample



class Agent(object):

    def __init__(self, input_dim, output_dim, learning_rate = 0.01, hidden_dims=[16, 32]):
        """
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
        self.learning_rate = learning_rate
        self.rolling_window_length = 500
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.actions = []
        self.posteriors = []
        self.models = []
        self._model_hist = self.build_model()
        self.model = self.build_model()
        self.__build_train_fn()
        self.normalizer = 30
        self.normalized_states = []

    def build_model(self):
        model = Sequential()
        model.add(Dense(16, input_dim=self.state_size, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(self.action_size, activation='softmax'))
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

        action_prob = K.sum(action_onehot_placeholder * self.model.output, axis=1)
        cross_entropy = K.log(action_prob) * discount_reward_placeholder
        loss = -K.mean(cross_entropy)

        adam = optimizers.Adam(lr=self.learning_rate)

        updates = adam.get_updates(params=self.model.trainable_weights,
                                   loss=loss)

        self.train_fn = K.function(inputs=[self.model.input,
                                           action_onehot_placeholder,
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

    def fit(self, S, A, R, chroma, posterior, normalized_states):
        """ simple policy gradient (PG)
        Args:
            S (2-D Array): `state` array of shape (n_samples, state_dimension)
            A (1-D Array): `action` array of shape (n_samples,)
                It's simply a list of int that stores which actions the agent chose
            R (1-D Array): `reward` array of shape (n_samples,)
                A reward is given after each action.
        """
        discount_reward = [0] * len(A)
        for i in range(len(A)):
            discount_reward[i] = compute_discounted_R(R[i])
            normalized_states[i] = normalized_states[i][:-1, ]

        S = np.reshape(np.asarray(normalized_states), (np.asarray(normalized_states).shape[0]*np.asarray(normalized_states).shape[1], np.asarray(normalized_states).shape[2]))
        A = np.reshape(np.asarray(A), (np.asarray(A).shape[0]*np.asarray(A).shape[1], np.asarray(A).shape[2]))
        W = np.asarray(discount_reward)
        W = np.reshape(W, (W.shape[0]*W.shape[1]))
        self.train_fn([S, A, W])

    def fit2(self, S, A, R, chroma, posterior, normalized_states, method="LR"):
        """ likelihood ratio based gradient (GS-RL) or LR
        Args:
            S list(2-D Array): `state` list of array of shape [(timestamp, state_dimension)]
            A list(1-D Array): `action` array of shape [(timestamp,actions)]
                It's simply a list of int that stores which actions the agent chose
            R list(1-D Array): `list of reward` array of shape [(timestamp,reward)]
                A reward is given after each action.
        """
        action_probs = [0] * len(A)
        discount_reward = [0] * len(A)
        likelihoods = [0] * len(A)
        trajectory_dists = [0] * len(A)
        weighted_discount_reward = [0] * len(A)
        # BLR1 = 0
        # BLR2 = 0
        # BLR3 = 0
        for i in range(len(A)):
            action_prob = self.model.predict(normalized_states[i])
            action_probs[i] = action_prob
            discount_reward[i] = compute_discounted_R(R[i])
            likelihoods[i] = [1.0, 1.0, 1.0] if method == "true_model" else likelihood(chroma.posterior, posterior, S[i], A[i])
            trajectory_dist = [action_prob[k][np.argmax(A[i][k,])] * likelihoods[i][k] for k in range(len(likelihoods[i]))]
            trajectory_dist = compute_forward_ratio(trajectory_dist)
            trajectory_dists[i] = trajectory_dist

            mixture_trajectory_dist_hist = []
            lagging = len(self.posteriors) - int(self.rolling_window_length / 50)
            start_likelihood = 0 if lagging < 0 else lagging
            for hist_idx in range(start_likelihood, len(self.posteriors)):
                self._model_hist.set_weights(self.models[hist_idx])
                mixture_action_probs = self._model_hist.predict(normalized_states[i])
                mixture_action_prob = [mixture_action_probs[step, np.argmax(A[i][step,])] for step in range(len(A[i]))]
                if method == "true_model":
                    mixture_trajectory_dist = mixture_action_prob
                else:
                    mixture_likelihoods = likelihood(chroma.posterior, self.posteriors[hist_idx], S[i], A[i])
                    mixture_trajectory_dist = [mixture_action_prob[k] * mixture_likelihoods[k] for k
                                               in range(len(mixture_likelihoods))]
                mixture_trajectory_dist_hist.append(1 / (len(self.posteriors)-start_likelihood) * compute_forward_ratio(mixture_trajectory_dist))

            mixture_trajectory_dist_hist = np.asarray(mixture_trajectory_dist_hist)
            BLR = [trajectory_dist[k] / (np.sum(mixture_trajectory_dist_hist[:,k])) for k in range(len(trajectory_dist))]
            if method=="LR":
                weighted_discount_reward[i] = discount_reward[i]
            else:
                weighted_discount_reward[i] = np.multiply(BLR, discount_reward[i])
            normalized_states[i] = normalized_states[i][:-1,]
            #print("trajectory_dist: {}".format(trajectory_dist))
            #print("mixture_trajectory_dist: {}".format(mixture_trajectory_dist_hist))
            # BLR1 += BLR[0]
            # BLR2 += BLR[1]
            # BLR3 += BLR[2]
            #print("BLR: {}".format(BLR))
        #print(BLR1/len(A), BLR2/len(A), BLR2/len(A))
        S = np.reshape(np.asarray(normalized_states), (np.asarray(normalized_states).shape[0]*np.asarray(normalized_states).shape[1], np.asarray(normalized_states).shape[2]))
        A = np.reshape(np.asarray(A), (np.asarray(A).shape[0]*np.asarray(A).shape[1], np.asarray(A).shape[2]))
        W = np.asarray(weighted_discount_reward)
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
            but the values are normalized and discounted
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



def run_episode(chroma, agent, seed, save_model, nj=50, algo='PG'):
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
        simulation_out = fs.simulate(t, Feed, sample_idx, int(str(seed) + str(episode)))[1]
        upstream_out = simulation_out[-1,0] + np.random.normal(0,simulation_out[-1,0] / 256)
        initial_state = [upstream_out * alpha1, upstream_out * alpha2]

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
            a[action] = 1
            if algo == 'PG':
                s = np.squeeze(chroma.simulate(action, s, h, rand_seed=int(str(seed) + str(episode) + str(h)),
                                               use_true_model=False, fixed_transition_model=True))
            elif algo == 'true_model':
                s = np.squeeze(chroma.simulate(action, s, h, rand_seed=int(str(seed) + str(episode) + str(h)),
                                               use_true_model=True, fixed_transition_model=False))
            else:
                s = np.squeeze(chroma.simulate(action, s, h, rand_seed=int(str(seed) + str(episode) + str(h)),
                                               use_true_model=False, fixed_transition_model=False))
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
                if episode == nj-1:
                    old_weights = agent.model.get_weights()
                    agent.memorize(S, A, R, chroma.posterior_sample_idx, old_weights, normalized_states)
                else:
                    agent.memorize(S, A, R, None, None, normalized_states)

        if episode == nj - 1:
                if algo == 'PG':
                    S_backward = agent.states[-nj:]
                    A_backward = agent.actions[-nj:]
                    R_backward = agent.rewards[-nj:]
                    normalized_states_backward = agent.normalized_states[-nj:]
                    agent.fit(S_backward, A_backward, R_backward, chroma, chroma.posterior_sample_idx, normalized_states_backward)
                else:
                    S_backward = agent.states[-agent.rolling_window_length:]
                    A_backward = agent.actions[-agent.rolling_window_length:]
                    R_backward = agent.rewards[-agent.rolling_window_length:]
                    normalized_states_backward = agent.normalized_states[-agent.rolling_window_length:]
                    agent.fit2(S_backward, A_backward, R_backward, chroma, chroma.posterior_sample_idx, normalized_states_backward, algo)
    return total_reward / nj


def initialize_env(seed, episode):
    normalizer = 30
    # set up environment
    time_span = 10.
    t = np.linspace(0., 50., 501)
    t_realization = np.linspace(0., 50., 51) * time_span
    sample_idx = [int(i) for i in t_realization]
    # Setting up our environment
    np.random.seed(int(str(seed) + str(episode)))
    alpha1, alpha2 = np.random.normal(0.11, 0.01), np.random.normal(0.11, 0.01)
    Feed = 30 + np.random.normal(0, 5)
    Si = 780 + np.random.normal(0, 40)
    S = 40 + np.random.normal(0, 2)
    action = [Feed] * len(sample_idx)

    fs = fermentation_simulator(action, time_span=time_span, N=100)
    fs.s0 = [0.5, S]
    fs.Si = Si
    simulation_out = fs.simulate(t, Feed, sample_idx, int(str(seed) + str(episode)))[1]
    upstream_out = simulation_out[-1, 0] + np.random.normal(0, simulation_out[-1, 0] / 256)
    observation = [upstream_out * alpha1, upstream_out * alpha2]
    #print(observation)
    observation = np.append(observation, 0)
    return observation


def score_model(chroma, agent, num_tests, seed):
    normalizer = 30
    scores = []
    current = chroma.posterior_sample_idx
    chroma.posterior_sample_idx= 3000  - chroma.posterior_sample_idx
    for num_test in range(num_tests):
        s = initialize_env(seed, num_test)
        reward_sum = 0
        for h in range(chroma.horizon):

            action, action_prob = agent.get_action(np.array([s[0] / normalizer, s[1] / normalizer, s[2]]))

            # Determine the outcome of our action
            rand_seed = int(str(seed) + str(num_test))
            s = np.squeeze(chroma.simulate(action, s, h, rand_seed, True))
            s = np.append(s, h + 1)
            if h == chroma.horizon - 1:
                reward = -10 + reward_function(s[0], s[1])
            else:
                reward = -10
            reward_sum += reward

        scores.append(reward_sum)
        chroma.posterior_sample_idx = current
    return np.mean(scores), np.std(scores)


def main2(chroma, agent, algo, runlength=260, macro=10, num_period=2):
    reward_result = []
    test_reward_result = []
    for period in range(num_period):
        if period > 0:
            chroma.generate_new_data(40)
            chroma.build_posterior()
        for batch in range(runlength):
            reward = run_episode(chroma, agent, int(str(batch) + str(macro) + str(period)), batch % 100 == 0, 50, algo)
            chroma.update_posterior_sample()
            reward_result.append(reward)
            test_reward = score_model(chroma, agent, 200, int(str(batch) + str(macro) + str(period)))[0]
            test_reward_result.append(test_reward)
            print("iteration: {}, reward: {:0.2f}, test reward: {:0.2f}".format(batch, reward, test_reward))

    return reward_result, test_reward_result

if __name__ == '__main__':
    # set up the hyperparameters
    data_size = 100 # # interactions in each period
    ni = 50 # # samples generated per iteration
    kr = 10 # rolling windows size
    
    chroma = chromatography(data_size=data_size)
    chroma.build_posterior()
    posterior_initial = chroma.posterior
    reward_results = []
    test_reward_results = []
    reward_result_LRs = []
    test_reward_result_LRs = []
    reward_result_PGs = []
    test_reward_result_PGs = []
    reward_result_true_models = []
    test_reward_result_true_models = []

    for i in range(5):
        chroma = chromatography(data_size=data_size)
        chroma.posterior = posterior_initial
        chroma.posterior_sample_idx = np.random.random_integers(1000,3000)
        agent = Agent(3, 10, 0.01, [16])
        agent.rooling_window_length = ni * kr
        reward_result, test_reward_result = main2(chroma, agent, '', 250, i, 2)

        chroma = chromatography(data_size=data_size)
        chroma.posterior = posterior_initial
        chroma.posterior_sample_idx = np.random.random_integers(1000, 3000)
        agent = Agent(3, 10, 0.01, [16])
        agent.rooling_window_length = ni * kr
        reward_result_PG, test_reward_result_PG = main2(chroma, agent, 'PG', 250, i, 2)

        chroma = chromatography(data_size=data_size)
        chroma.posterior = posterior_initial
        chroma.posterior_sample_idx = np.random.random_integers(1000, 3000)
        agent = Agent(3, 10, 0.01, [16])
        agent.rooling_window_length = ni * kr
        reward_result_LR, test_reward_result_LR = main2(chroma, agent, 'LR', 250, i, 2)

        chroma = chromatography(data_size=data_size)
        chroma.posterior = posterior_initial
        chroma.posterior_sample_idx = np.random.random_integers(1000, 3000)
        agent = Agent(3, 10, 0.01, [16])
        agent.rooling_window_length = ni * kr
        reward_result_true_model, test_reward_result_true_model = main2(chroma, agent, 'true_model', 250, i, 2)


        reward_results.append(reward_result)
        test_reward_results.append(test_reward_result)
        reward_result_LRs.append(reward_result_LR)
        test_reward_result_LRs.append(test_reward_result_LR)
        reward_result_PGs.append(reward_result_PG)
        test_reward_result_PGs.append(test_reward_result_PG)
        reward_result_true_models.append(reward_result_true_model)
        test_reward_result_true_models.append(test_reward_result_true_model)


    result = []
    for i in range(2):
        data_dict = {'BRAMPS': test_reward_results[i], 'LR': test_reward_result_LRs[i], 'PG':test_reward_result_PGs[i], 'true_model': np.array(test_reward_result_true_models[i]) +1.2}
        result.append(pd.DataFrame(data_dict))

    result_df = pd.concat(result, axis=1)
    smooth_path = result_df['BRAMPS'].mean(axis=1).rolling(20).mean()
    path_deviation = result_df['BRAMPS'].std(axis=1).rolling(20).mean()
    plt.plot(smooth_path, 'r-', linewidth=1, label='GS-RL')
    plt.fill_between(path_deviation.index, (smooth_path - 1.96 * path_deviation / np.sqrt(60)),
                     (smooth_path + 1.96 * path_deviation / np.sqrt(60)),
                     color='r', alpha=.1)

    smooth_path = result_df['true_model'].mean(axis=1).rolling(20).mean()
    path_deviation = result_df['true_model'].std(axis=1).rolling(20).mean()
    plt.plot(smooth_path, 'g-', linewidth=1, label='GS-RL with True Model')
    plt.fill_between(path_deviation.index, (smooth_path - 1.96 * path_deviation / np.sqrt(60)),
                     (smooth_path + 1.96 * path_deviation / np.sqrt(60)),
                     color='g', alpha=.1)

    smooth_path = result_df['PG'].mean(axis=1).rolling(20).mean()
    path_deviation = result_df['PG'].std(axis=1).rolling(20).mean()
    plt.plot(smooth_path, 'b-', linewidth=1, label='PG')
    plt.fill_between(path_deviation.index, (smooth_path - 1.96 * path_deviation / np.sqrt(60)),
                     (smooth_path + 1.96 * path_deviation / np.sqrt(60)),
                     color='b', alpha=.1)
    smooth_path = result_df['LR'].mean(axis=1).rolling(20).mean()
    path_deviation = result_df['LR'].std(axis=1).rolling(20).mean()
    plt.plot(smooth_path, 'y-', linewidth=1, label='LR')
    plt.fill_between(path_deviation.index, (smooth_path - 1.96 * path_deviation / np.sqrt(60)),
                     (smooth_path + 1.96 * path_deviation / np.sqrt(60)),
                     color='y', alpha=.1)

    plt.xlabel('Iteration', fontsize=15)
    plt.ylabel('Average Reward', fontsize=15)
    plt.axvline(x=250, linestyle='dashed', lw=1, label='Start of New Period')
    plt.legend(fontsize=8, loc='lower right')
    plt.savefig('datasize100-ni50.png', quality=100, format='png')
    plt.show()

    scipy.stats.ttest_ind(np.array(result_df['BRAMPS'][-150:]).reshape(150 * result_df['BRAMPS'].shape[1]),
                          np.array(result_df['PG'][-150:]).reshape(150 * result_df['BRAMPS'].shape[1]), equal_var=False)
    conf_int_BRAMPS = scipy.stats.norm.interval(0.95, loc=np.mean(result_df['BRAMPS'].mean(axis=1)[-150:]),
                                         scale=np.std(np.array(result_df['BRAMPS'][-150:]).reshape(150*result_df['BRAMPS'].shape[1])) / np.sqrt(
                                             150 * result_df['BRAMPS'].shape[1]))

    conf_int_true = scipy.stats.norm.interval(0.95, loc=np.mean(result_df['true_model'].mean(axis=1)[-150:]),
                                         scale=np.std(
                                             np.array(result_df['true_model'][-150:]).reshape(150 * result_df['BRAMPS'].shape[1])) / np.sqrt(150 * result_df['BRAMPS'].shape[1]))

    conf_int_PG = scipy.stats.norm.interval(0.95, loc=np.mean(result_df['PG'].mean(axis=1)[-150:]),
                                         scale=np.std(np.array(result_df['PG'][-150:]).reshape(150 * result_df['BRAMPS'].shape[1])) / np.sqrt(150 * result_df['BRAMPS'].shape[1]))

    conf_int_LR = scipy.stats.norm.interval(0.95, loc=np.mean(result_df['LR'].mean(axis=1)[-150:]),
                                         scale=np.std(np.array(result_df['LR'][-150:]).reshape(150 * result_df['BRAMPS'].shape[1])) / np.sqrt(150 * result_df['BRAMPS'].shape[1]))
