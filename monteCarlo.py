import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean

def gen_simulation(payoff_type, runs, edge, starting_cap):
    mu=0
    sigma=1
    rs = np.random.RandomState(seed=42)
    dist = rs.normal(mu, sigma, 1000)
    exp_concave=[-(j**2) for j in dist]
    exp_concave = mean(exp_concave)

    exp_convex = [j**2 for j in dist]
    exp_convex=mean(exp_convex)

    sim_list=[]
    i = 0
    while i < runs:
        if payoff_type == 'concave':
            for row in dist:
                sim_list.append(exp_concave*(1+edge)-row**2)
            i += 1
        else:
            for row in dist:
                sim_list.append(-exp_convex*(1-edge)+row**2)
            i += 1
    sim_list=np.cumsum(sim_list)
    return sim_list

if __name__ == '__main__':
    x = gen_simulation('convex', 100, 0.05)
    plt.plot(x)
    plt.show()