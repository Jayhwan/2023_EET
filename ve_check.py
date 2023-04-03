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
            }
    game_se = EETGame(load, pv, eet_input=eet_input, hyper_input=hyper_input)
    s1 = time.time()
    game_se.followers_ve_iter(20, indiv=False)
    e1 = time.time()
    game_se.is_ve()

    eet_input = {'total_users': n,
                 'active_users': a,
                 'time_horizon': t, }
    hyper_input = {'ve_step_size_follower': 0.01,
                   }
    game_se = EETGame(load, pv, eet_input=eet_input, hyper_input=hyper_input)
    s2 = time.time()
    game_se.followers_ve_iter(10, indiv=True)
    e2 = time.time()
    game_se.is_ve()

    print(f"Together : {(e1-s1)/50}, Indiv : {(e2-s2)/5}")
