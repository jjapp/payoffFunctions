from scipy.integrate import simps
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def get_gain_function(dist, payoff_type, edge, cutoff=-30, f_max=0.06):
    payoff = []
    # concave vs convex:
    if payoff_type == 'concave':
        # get the square of each number in the dist
        new_dist = [number**2 for number in dist]

        # get the expectation of the new distribution
        expected = mean(new_dist)
        alpha = (1 + edge) * expected

        for row in new_dist:
            s = alpha - row
            payoff.append(s)

    else:
        # get the square of each number in the dist
        new_dist = [number**2 for number in dist]

        expected = mean(new_dist)
        alpha = expected * (1 - edge)

        for row in new_dist:
            s = row - alpha
            payoff.append(s)
    # cap the payoff function
    payoff = filter(lambda z: z > cutoff, payoff)
    payoff = list(payoff)

    f = np.linspace(0, f_max, 100)

    # get gain function
    g_list = []
    for i in f:
        int_vector = []
        for j in payoff:
            obs = np.log(1 + i * j)
            int_vector.append(obs)
        gain = simps(int_vector)/(len(dist))
        g_list.append(gain)

    return (f, g_list)


def plot_gain_function(gain1, gain2, label1, label2):
    plt.plot(gain1[0], gain1[1], label=label1)
    plt.plot(gain2[0], gain2[1], label=label2)
    plt.title("Gain function for Concave and Convex Bets")
    plt.legend()
    plt.savefig('gainFunctions.png')
    plt.show()


if __name__ == '__main__':
    random_state=np.random.RandomState(seed=42)
    mu = 0
    sigma = 1
    s = random_state.normal(mu, sigma, 1000)
    x1 = get_gain_function(s, 'convex', 0.05)
    x2 = get_gain_function(s, 'concave', 0.05)
    plot_gain_function(x1, x2, "Convex", "Concave")


