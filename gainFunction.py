from scipy.integrate import simps
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

def get_gain_function(dist, payoff_type, edge, cutoff=-3):
    payoff = []
    # concave vs convex:
    if payoff_type == 'concave':
        new_dist = filter(lambda x: x > cutoff, dist)
        new_dist = list(new_dist)
        expected = mean(new_dist)
        alpha = (1 + edge) * expected
        lower_bound = cutoff
        f = np.linspace(0, -1 / cutoff, 100)
        for row in new_dist:
            s = alpha - row ** 2
            payoff.append(s)

    else:
        expected = mean(dist)
        alpha = (1 + edge) * expected
        f = np.linspace(0, 1 / alpha, 100)
        for row in dist:
            s = row ** 2 - alpha
            payoff.append(s)

    # get gain function
    g_list = []
    for i in f:
        int_vector = []
        for j in new_dist:
            obs = np.log(1 + i * j)
            int_vector.append(obs)
        gain = simps(int_vector, new_dist)
        g_list.append(gain)

    return (f, g_list)


def plot_gain_function(f, g_list):
    plt.plot(f, g_list)
    plt.show()
