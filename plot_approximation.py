import matplotlib.pyplot as plt

def plot_approximation(X, Y, X_sample, Y_sample, std):
    plt.fill_between(X.ravel(), Y[:, 0].ravel() + 1.96 * std, Y[:, 0].ravel() - 1.96 * std, alpha=0.1)
    plt.fill_between(X.ravel(), Y[:, 1].ravel() + 1.96 * std, Y[:, 1].ravel() - 1.96 * std, alpha=0.1)
    plt.plot(X, Y[:,0], 'b-', lw=3, label='True bioreactor biomass trajectory')
    plt.plot(X, Y[:,1], 'r-', lw=3, label='True bioreactor substrate trajectory')
    plt.xlabel('time', fontsize=10)
    # plt.plot(X, mu, 'b-', lw=3, label='GP Surrogate function')
    plt.plot(X_sample, Y_sample[:,0], 'kx', mew=3,ms=3, label='Noisy samples of biomass')
    plt.plot(X_sample, Y_sample[:,1], 'ko', mew=3, ms=3, label='Noisy samples of substrate')
    plt.legend(fontsize=5)
