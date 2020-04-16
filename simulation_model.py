import numpy as np
import scipy
from scipy.integrate import odeint
from plot_approximation import plot_approximation
import matplotlib.pyplot as plt
from dynesty.utils import resample_equal
from dynesty import NestedSampler, DynamicNestedSampler
from dynesty import plotting as dyplot
import copy
import pymc3 as pm
import arviz as az
from scipy.stats import beta

'''
# bioreactor differential equations
# @param: Ks, q_s_max, Y_em, q_m
'''

class fermentation_simulator:
    def __init__(self,
                 action,
                 time_span,
                 std=5,
                 N=50,
                 Y_em = 0.3,
                 q_s_max = 0.57,
                 q_m = 0.013,
                 Ks = 0.1):
        # define parameters in bioreactor model
        self.N = N
        self.std = std
        self.action = action
        self.time_span = time_span
        self.Y_em = Y_em # [0.1,0.9]
        self.q_s_max = q_s_max # [0.2,0.8]
        self.q_m = q_m # [0,0.05]
        self.Ks = Ks
        # define initial state, [biomass, substrate]
        self.s0 = [0.5, 40]
        self.V = 1000
        self.Si = 780
        self.LN2PI = np.log(2. * np.pi)
        self.X = None
        self.S = None

    def feed_rate(self, t):

        return self.action[int(t / self.time_span)]

    def bioreactor(self, z, t, feed):
        X, S = z
        # rate of change
        q_s = self.q_s_max * S / (S + self.Ks)
        mu = (q_s - self.q_m) * self.Y_em

        dX = (-self.action[int(t / self.time_span)] / self.V + mu) * X
        dS = self.feed_rate(t) / self.V * (self.Si - S) - q_s * X
        return [dX, dS]

    def simulate(self, t, feed, sample_idx):
        true_trajectories = scipy.integrate.odeint(self.bioreactor, self.s0, t, args=(feed,))
        substrate = true_trajectories[sample_idx, 1] + np.random.normal(0, self.std, true_trajectories[sample_idx, 1].shape)
        biomass = true_trajectories[sample_idx, 0] + np.random.normal(0, self.std, true_trajectories[sample_idx, 1].shape)
        substrate[substrate<0] = 0.
        biomass[biomass<0] = 0.
        sample_trajectories = np.transpose(np.array([biomass,substrate]))
        return true_trajectories, sample_trajectories

    def posterior_sampler(self, t, sample_idx, ndims=2):
        nlive = 1024  # number of (initial) live points
        bound = 'multi'  # use MutliNest algorithm
        sample = 'rwalk'
        self.X = np.zeros(shape=(self.N, len(sample_idx)))
        self.S = np.zeros(shape=(self.N, len(sample_idx)))
        for i in range(self.N):
            _, sample_trajectories = self.simulate(t, self.action, sample_idx)
            self.X[i,:] = sample_trajectories[:,0]
            self.S[i,:] = sample_trajectories[:,1]
        dsampler = DynamicNestedSampler(self.loglikelihood, self.prior_transform, ndims, bound=bound)
        dsampler.run_nested(nlive_init=nlive)
        res = dsampler.results
        weights = np.exp(res['logwt'] - res['logz'][-1])
        samples_dynesty = resample_equal(res.samples, weights)
        return dsampler


    def plot(self, t, feed, sample_idx):
        # simulate the system for different values of feed rate.
        [true_trajectories, sample_trajectories] = self.simulate(t, feed,  sample_idx)
        # plot the trajectories.
        plot_approximation(t, true_trajectories, t[sample_idx], sample_trajectories, self.std)

    def loglikelihood(self, theta):
        Y_em, q_s_max = theta
        loglike = 0
        for i in range(self.N):
            loglike += self.loglikelihood_single_data(self.action, Y_em, q_s_max, self.X[i,:], self.S[i,:], self.time_span)

        return loglike

    def loglikelihood_single_data(self, action, Y_em, q_s_max,  X, S, time_span):
        epsilon = 0.1
        horizon = len(S)
        q_m = self.q_m
        # rate of change
        for i in range(horizon):
            mean_S = (S[i+1] + S[i])/2
            mean_X = (X[i+1] + X[i])/2
            increment_X = X[i+1] - X[i]
            increment_S = S[i + 1] - S[i]
            feed = action[i]

            q_s = q_s_max * mean_S / (mean_S + self.Ks)
            mu = (q_s - q_m) * Y_em

            muX = (-feed/self.V + mu) * mean_X * time_span
            sigmaX = (feed / self.V + abs(mu)) * np.abs(mean_X) * time_span
            muS = (feed/self.V * (self.Si - mean_S) - q_s * mean_X) * time_span
            sigmaS = (feed/self.V * (self.Si + abs(mean_S)) + q_s * abs(mean_X)) * time_span
            sigmaX = epsilon if sigmaX == 0 else sigmaX
            sigmaS = epsilon if sigmaS == 0 else sigmaS
            return self.loglikelihood_single_normal(muX,sigmaX,increment_X) + self.loglikelihood_single_normal(muS,sigmaS,increment_S)

    def loglikelihood_single_normal(self, mu, sigma, x):
        """
        The log-likelihood function for single normal
        """
        # normalisation
        LNSIGMA = np.log(sigma)
        norm = -0.5 * self.LN2PI - LNSIGMA
        # chi-squared (data, sigma and x are global variables defined early on in this notebook)
        chisq = ((x - mu) / sigma) ** 2

        return norm - 0.5 * chisq

    def prior_transform(self, theta):
        """
        A function defining the tranform between the parameterisation in the unit hypercube
        to the true parameters.

        Args:
            theta (tuple): a tuple containing the parameters.

        Returns:
            tuple: a new tuple or array with the transformed parameters.
        """

        Y_em, q_s_max = theta

        Y_em_min = 0.1  # lower bound on uniform prior on c
        Y_em_max = 0.5  # upper bound on uniform prior on c
        q_s_max_min = 0.3  # lower bound on uniform prior on c
        q_s_max_max = 0.8  # upper bound on uniform prior on c
        #q_m_min = -10.  # lower bound on uniform prior on c
        #q_m_max = 10.  # upper bound on uniform prior on c
        Y_em = Y_em * (Y_em_max - Y_em_min) + Y_em_min  # convert back to c
        q_s_max = q_s_max * (q_s_max_max - q_s_max_min) + q_s_max_min  # convert back to c
        #q_m = q_m * (q_m_max - q_m_min) + q_m_min  # convert back to c

        return Y_em, q_s_max #, q_m



class chromatography:
    def __init__(self,
                 data_size=200,
                 horizon=3,
                 N=50):
        # define parameters in bioreactor model
        purity_coef = [0.48, 0.28]
        self.chrom1_protein = [self.get_alpha_beta(mu=purity, sigma=0.08 * purity_coef[0]) for purity in [(1 + pool / 12) * purity_coef[0] for pool in range(10)]]
        self.chrom1_impurity = [self.get_alpha_beta(mu=purity, sigma=0.08 * purity_coef[1]) for purity in [(1 + pool / 12) * purity_coef[1] for pool in range(10)]]

        #self.chrom1_protein = [[purity - 0.1 * purity_coef[0], purity + 0.1 * purity_coef[0]] for purity in [(1 + pool / 11) * purity_coef[0] for pool in range(10)]]
        #self.chrom1_impurity = [[purity - 0.1 * purity_coef[1], purity + 0.1 * purity_coef[1]] for purity in [(1 + pool / 11) * purity_coef[1] for pool in range(10)]]

        purity_coef = [0.5, 0.25]
        self.chrom2_protein = [self.get_alpha_beta(mu=purity, sigma=0.08 * purity_coef[0]) for purity in [(1 + pool / 12) * purity_coef[0] for pool in range(10)]]
        self.chrom2_impurity = [self.get_alpha_beta(mu=purity, sigma=0.08 * purity_coef[1]) for purity in [(1 + pool / 12) * purity_coef[1] for pool in range(10)]]

        purity_coef = [0.5, 0.2]
        self.chrom3_protein = [self.get_alpha_beta(mu=purity, sigma=0.08 * purity_coef[0]) for purity in [(1 + pool / 12) * purity_coef[0] for pool in range(10)]]
        self.chrom3_impurity = [self.get_alpha_beta(mu=purity, sigma=0.08 * purity_coef[1]) for purity in [(1 + pool / 12) * purity_coef[1] for pool in range(10)]]

        self.true_model_params = {1:{'protein': self.chrom1_protein, 'impurity': self.chrom1_impurity},
                             2: {'protein': self.chrom2_protein, 'impurity': self.chrom2_impurity},
                             3: {'protein': self.chrom3_protein, 'impurity': self.chrom3_impurity}}

        self.sim_size = 10
        self.horizon = horizon
        self.posterior = {}
        self.posterior_pareto = {}
        self.N = N
        self.data_state, self.data_action = self.generate_data(data_size)
        self.k=3
        self.leftover_sample = {} # self.cur_sample_idx[step][0][window]
        self.posterior_sample_idx = 0

    def get_alpha_beta(self, alpha=None, beta=None, mu=None, sigma=None):
        if (alpha is not None) and (beta is not None):
            pass
        elif (mu is not None) and (sigma is not None):
            kappa = mu * (1 - mu) / sigma**2 - 1
            alpha = mu * kappa
            beta = (1 - mu) * kappa
        else:
            raise ValueError('Incompatible parameterization. Either use alpha '
                             'and beta, or mu and sigma to specify distribution.')

        return alpha, beta

    def posterior_generator(self, data, size=1000):
        # data = np.random.beta(5, 6, 100)
        with pm.Model() as model_g:
            # alpha = pm.TruncatedNormal(mu=3,sigma=2,lower=0.1,name='alpha')
            # beta = pm.TruncatedNormal(mu=7,sigma=2,lower=0.1,name='beta')
            alpha = pm.Uniform(lower=0, upper=100, name='alpha')
            beta = pm.Uniform(lower=0, upper=100, name='beta')
            y = pm.Beta('y', alpha=alpha, beta=beta, observed=data)
            trace_g = pm.sample(size, tune=1000, cores=4)
            #az.summary(trace_g)
        return np.array([trace_g.get_values('alpha'), trace_g.get_values('beta')]) # 2 * size


    def simulate(self, window, initial_state, step, rand_seed=None):
        if self.posterior_sample_idx >= 4000:
            return
        if rand_seed == None:
            np.random.seed(int(str(window) + str(step) + str(int((initial_state[0] - int(initial_state[0])) * 10**6))))
        else:
            np.random.seed(rand_seed)
        protein_param = self.posterior[step][0][window][:, self.posterior_sample_idx]
        impurity_param = self.posterior[step][1][window][:, self.posterior_sample_idx]

        removal_rate_protein = np.random.beta(protein_param[0], protein_param[1])
        removal_rate_impurity = np.random.beta(impurity_param[0], impurity_param[1])

        protein = initial_state[0] * removal_rate_protein
        impurity = initial_state[1] * removal_rate_impurity
        state = [protein, impurity]
        return state

    def build_posterior(self):
        for i in range( self.horizon):
            actions = self.data_action[:, i]
            unique_actions = np.unique(actions)
            for j in [0, 1]:
                state = self.data_state[:,i,j]
                next_state = self.data_state[:,i+1,j]
                for a in unique_actions:
                    action_index = actions == a
                    data = next_state[action_index] / state[action_index]
                    posteriors = self.posterior_generator(data)
                    if i in self.posterior:
                        if j in self.posterior[i]:
                            self.posterior[i][j][a] = posteriors
                            self.leftover_sample[i][j][a] = posteriors.shape[1] - 1
                        else:
                            self.posterior[i][j] = {a:posteriors}
                            self.leftover_sample[i][j] = {a: posteriors.shape[1] - 1}
                    else:
                        self.posterior[i] = {j:{a:posteriors}}
                        self.leftover_sample[i] = {j: {a: posteriors.shape[1] - 1}}

    def update_posterior_sample(self):
        self.posterior_sample_idx += 1

    def generate_data(self, data_size=1000):
        time_span = 10.
        t = np.linspace(0., 50., 501)
        t_realization = np.linspace(0., 50., 51) * time_span
        sample_idx = [int(i) for i in t_realization]
        data_state = []
        data_action = []
        for i in range(data_size):
            alpha1, alpha2 = np.random.normal(0.11, 0.01), np.random.normal(0.11, 0.01)
            Feed = 30 + np.random.normal(0, 5)
            Si = 780 + np.random.normal(0, 40)
            S = 40 + np.random.normal(0, 2)
            action = [Feed] * len(sample_idx)
            fs = fermentation_simulator(action, time_span=time_span, N=100)
            fs.s0 = [0.5, S]
            fs.Si = Si
            simulation_out = fs.simulate(t, Feed, sample_idx)[1]
            upstream_out = simulation_out[-1, 0] + np.random.normal(0, simulation_out[-1, 0] / 256)
            state = [upstream_out * alpha1, upstream_out * alpha2]
            S = [state]
            A = []
            for h in range(self.horizon):
                action = int(np.random.uniform(0, 10))
                protein_model_param = self.true_model_params[h+1]['protein'][action]
                protein_removal_rate = np.random.beta(a=protein_model_param[0], b=protein_model_param[1])
                impurity_model_param = self.true_model_params[h+1]['impurity'][action]
                impurity_removal_rate = np.random.beta(a=impurity_model_param[0], b=impurity_model_param[1])
                state = [state[0] * protein_removal_rate, state[1] * impurity_removal_rate]
                S.append(state)
                A.append(action)
            data_state.append(S)
            data_action.append(A)
        return np.array(data_state), np.array(data_action)


if __name__ == '__main__':
    feed = 10
    time_span = 10.
    t = np.linspace(0., 50., 501)
    t_realization = np.linspace(0., 50., 51) * time_span
    sample_idx = [int(i) for i in t_realization]
    std = 10.
    action = [feed] * len(sample_idx)


    fs = fermentation_simulator(action, time_span,N=100)
    simulation_out = fs.simulate(t, feed, sample_idx)[0]

    fs.plot(t,feed,sample_idx)
    plt.show()

    result = fs.posterior_sampler(t, sample_idx, 2)
    res = result.results
    weights = np.exp(res['logwt'] - res['logz'][-1])
    samples_dynesty = resample_equal(res.samples, weights)
    fig, axes = dyplot.traceplot(result.results, truths=[0.3, 0.57,0.013],
                                 show_titles=True, trace_cmap='viridis',
                                 quantiles=None)
    plt.show()

    chroma = chromatography()
    _, simulated_data = fs.simulate(t, feed, sample_idx)
    initial_state = np.array([[simulated_data[-1, 0]*0.5, simulated_data[-1, 0] * 0.4], [simulated_data[-1, 0]*0.45, simulated_data[-1, 0]* 0.39]])
    initial_state = np.array([[simulated_data[-1, 0]*0.5, simulated_data[-1, 0] * 0.4]])

    chroma.simulate([1,2], 2,initial_state, False)
    chroma.build_posterior()
    chroma.postreior_sampler()
    a,b = chroma.simulate([1, 2], 1, initial_state, True)
    chroma.likelihood(np.array([[initial_state[0,0],initial_state[0,1]], [a[0,0],b[0,0]], [a[0,1],  b[0,1]]]), [1,2])
    chroma.single_step_likelihood([a[0,1],  b[0,1]], [a[0,0],b[0,0]], 2,2)


    chroma = chromatography()
    chroma.build_posterior()
    chroma.simulate(window=5, initial_state=[20,19], step=0)
