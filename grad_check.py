from Prob_EET import HyperParameters, EETParameters, Leader, Follower, EETGame
import numpy as np
import cvxpy as cp
import pickle
import matplotlib.pyplot as plt
import time


if __name__=="__main__":
    n = 4
    a = 2
    t = 3
    load = np.load("./data/load_123.npy", allow_pickle=True)[:n, :t]
    pv = np.load("./data/E_PV.npy", allow_pickle=True)[:t]

    eet_input = {'total_users': n,
                 'active_users': a,
                 'time_horizon': t, }
    hyper_input = { 've_step_size_follower': 0.01,
                    'grad_step_size':0.0001
            }
    game_se = EETGame(load, pv, eet_input=eet_input, hyper_input=hyper_input)
    s1 = time.time()
    game_se.followers_ve_iter(20, indiv=False)
    e1 = time.time()
    game_se.is_ve()

    print(f"Together : {(e1-s1)/20}")

    g_s, g_b = game_se.compute_leader_gradient()

    np.set_printoptions(precision=3, suppress=True)
    print(g_s.shape, g_b.shape)
    print(g_s, g_b)
