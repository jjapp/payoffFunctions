# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from gainFunction import get_gain_function
import matplotlib.pyplot as plt
from random import random, seed

# create the random distributions

def get_fat_tail(mu, sigma1, sigma2, n):
    '''Returns an np array of a cauchy distribution with n samples'''
    seed(42)
    random_state = np.random.RandomState(seed=42)
    dist = []
    i = 0
    while i < n:
        value = random()
        if value > 0.9:
            dist.append(random_state.normal(mu, sigma1))
            i += 1
        else:
            dist.append(random_state.normal(mu, sigma2))
            i += 1
    return dist
# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    d1 = get_fat_tail(0, 1, 3, 500)
    d2 = get_fat_tail(0, 1, 3, 1000)
    d3 = get_fat_tail(0, 1, 3, 5000)
    x1 = get_gain_function(d1, 'convex', 0.05, f_max=0.0075)
    x2 = get_gain_function(d2, 'convex', 0.05, f_max=0.0075)
    x3 = get_gain_function(d3, 'convex', 0.05, f_max=0.0075)

    plt.plot(x1[0], x1[1], label='100 Samples')
    plt.plot(x2[0], x2[1], label='500 Samples')
    plt.plot(x3[0], x3[1], label='1000 Samples')
    plt.title("Gain function at various sample sizes")
    plt.legend()
    plt.savefig('comparison.png')
    plt.show()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
