from Prob_EET import HyperParameters, EETParameters, Leader, Follower, EETGame
import numpy as np
import cvxpy as cp
import pickle
import matplotlib.pyplot as plt
import time


if __name__=="__main__":
    n = 100
    a = 50
    t = 24
    load = np.load("./data/load_123.npy", allow_pickle=True)[:n, :t]
    pv = np.load("./data/E_PV.npy", allow_pickle=True)[:t]

    eet_input = {'total_users': n,
                 'active_users': a,
                 'time_horizon': t, }
    hyper_input = { 've_step_size_follower': 0.01,
                    'grad_step_size':0.005,
                    'grad_max_iter':400,
                    'spne_max_iter':400,
                    've_max_iter': 20,
            }
    game_se = EETGame(load, pv, eet_input=eet_input, hyper_input=hyper_input)
    game_spne = EETGame(load, pv, eet_input=eet_input, hyper_input=hyper_input)


    diff1, par_hist1, ec_hist1, l_util_hist1 = game_se.se_algorithm(indiv=False)

    diff2, par_hist2, ec_hist2, l_util_hist2 = game_spne.spne_algorithm(fix_leader=True)

    plt.figure()
    plt.title('PAR')
    plt.plot(par_hist1, label='se')
    plt.plot(par_hist2, label='spne')
    plt.legend()
    plt.show()

    plt.figure()
    plt.title('Leader Utility')
    plt.plot(l_util_hist1, label='se')
    plt.plot(l_util_hist2, label='spne')
    plt.legend()
    plt.show()
