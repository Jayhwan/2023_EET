import numpy as np
import cvxpy as cp
import pickle
import matplotlib.pyplot as plt
import time


class HyperParameters:
    def __init__(self, hyper_input={}):
        self.grad_step_size = hyper_input.get('grad_step_size', 0.0005)
        self.grad_max_iter = hyper_input.get('grad_max_iter', 50)
        self.grad_eps = 1e-4

        self.ve_step_size_leader = 0.3
        self.ve_step_size_follower = hyper_input.get('ve_step_size_follower', .05)
        self.ve_max_iter = hyper_input.get('ve_max_iter', 20)

        self.ve_eps = 1.0e-10
        self.spne_max_iter = hyper_input.get('spne_max_iter', 50)
        self.active_epsilon = 1e-5

        self.prox_gamma = 1
        self.prox_eps = 1e-5
        self.prox_max_iter = 1000


class EETParameters:
    def __init__(self, eet_input={}):
        self.total_users = eet_input.get('total_users', 20)
        self.active_users = eet_input.get('active_users', 10)
        self.passive_users = self.total_users - self.active_users
        self.time_horizon = eet_input.get('time_horizon', 12)
        self.alpha = 0.9956
        self.beta_s = 0.99
        self.beta_b = 1.01

        self.p_soh = 0.01
        self.p_e = np.ones(self.time_horizon)
        self.p_tax = 0.001

        self.q_max = 1000.
        self.q_min = 0.
        self.q_init = 0.
        self.c_s = 200.
        self.c_b = 200.


class Leader:
    def __init__(self, initial_decision=None, time_horizon=24):
        # initial_decision = [self.sell, self.buy]
        self.time = time_horizon
        if initial_decision is None:
            self.sell = 0.5 * np.ones(self.time)
            self.buy = np.ones(self.time)
        else:
            self.sell = initial_decision[0]
            self.buy = initial_decision[1]

    def update_grad(self, gradient, step_size=0.5):
        self.sell += step_size * gradient[0]
        self.buy += step_size * gradient[1]
        p_s = cp.Variable(self.time, nonneg=True)
        p_b = cp.Variable(self.time, nonneg=True)
        obj = cp.Minimize(cp.sum(cp.power(p_s - self.sell, 2) + cp.sum(cp.power(p_b - self.buy, 2))))
        const = [p_s <= p_b]
        prob = cp.Problem(obj, const)
        result = prob.solve(solver='ECOS')
        self.sell = p_s.value
        self.buy = p_b.value

        return step_size * np.linalg.norm(np.concatenate([gradient[0], gradient[1]]))

    def update_direct(self, next_decision):
        [self.sell, self.buy] = next_decision


class Follower:
    def __init__(self, energy_use, is_active=True, time_horizon=24):
        self.usage = energy_use
        assert len(self.usage)==time_horizon
        self.active = is_active

        if not self.active:
            self.usage = np.maximum(self.usage, 0)

        self.load = np.maximum(self.usage, 0)
        self.sell = np.maximum(-self.usage, 0)
        self.buy = np.zeros(time_horizon)

    def update_direct(self, next_decision):
        if self.active:
            [self.sell, self.buy] = next_decision
            self.load = self.sell - self.buy + self.usage
        else:
            pass


class EETGame:
    def __init__(self, e_ha_matrix, e_pv_matrix, hyper_input=None, eet_input=None, filename=None):
        self.hyper_param = HyperParameters(hyper_input)
        self.eet_param = EETParameters(eet_input)

        self.leader = Leader(time_horizon=self.eet_param.time_horizon)
        self.active_followers = []
        self.passive_followers = []
        self.usage_matrix = e_ha_matrix - e_pv_matrix
        for i in range(self.eet_param.active_users):
            self.active_followers += [Follower(energy_use = self.usage_matrix[i], is_active=True, time_horizon=self.eet_param.time_horizon)]
        for i in range(self.eet_param.active_users, self.eet_param.total_users):
            self.passive_followers += [Follower(energy_use = self.usage_matrix[i], is_active=False, time_horizon=self.eet_param.time_horizon)]

        self.all_followers = [*self.active_followers, *self.passive_followers]

        self.filename = filename

    def leader_action(self):
        return self.leader.sell, self.leader.buy

    def followers_action(self):
        x_s = np.zeros((self.eet_param.total_users, self.eet_param.time_horizon))
        x_b = np.zeros((self.eet_param.total_users, self.eet_param.time_horizon))
        l = np.zeros((self.eet_param.total_users, self.eet_param.time_horizon))

        for i, follower in enumerate(self.active_followers):
            x_s[i] = follower.sell
            x_b[i] = follower.buy
            l[i] = follower.load
        x_s_active = x_s[:self.eet_param.active_users]
        x_b_active = x_b[:self.eet_param.active_users]
        l_active = l[:self.eet_param.active_users]

        for i , follower in enumerate(self.passive_followers, start=self.eet_param.active_users):
            x_s[i] = follower.sell
            x_b[i] = follower.buy
            l[i] = follower.load

        return x_s, x_b, l, x_s_active, x_b_active, l_active

    def update_followers(self, next_decisions):
        for i in range(self.eet_param.total_users):
            self.all_followers[i].update_direct(next_decisions[i])

    def info_func(self):
        x_s_all, x_b_all, l_all, x_s_active, x_b_active, l_active = self.followers_action()
        p_s = self.leader.sell
        p_b = self.leader.buy

        load = np.sum(l_all, axis=0)
        par = np.max(load)/np.average(load)
        ec = np.sum(np.multiply(self.eet_param.p_e, np.power(load, 2)))
        l_util = -np.sum(np.multiply(self.eet_param.p_e, np.power(load, 2)))-self.eet_param.p_tax*np.sum(np.power(p_s, 2) + np.power(p_b, 2))
        f_util_list = []
        for i in range(self.eet_param.total_users):
            f_util_list += [np.sum(np.multiply(p_s, x_s_all[i]) - np.multiply(p_b, x_b_all[i]))
                            - np.sum(np.multiply(self.eet_param.p_e, np.multiply(l_all[i], np.sum(l_all, axis=0))))
                            - self.eet_param.p_soh*(np.sum(np.multiply(x_s_all[i], np.sum(x_s_all, axis=0))
                                                           + np.multiply(x_b_all[i], np.sum(x_b_all, axis=0))))]

        return par, ec, l_util, f_util_list

    def followers_ve_iter(self, num_iter, indiv=False):
        print("Compute the followers VE")
        p_s, p_b = self.leader_action()
        diff = 0

        load_curve = []

        for iter in range(num_iter):
            diff = 0

            x_s_all, x_b_all, l_all, x_s_active, x_b_active, l_active = self.followers_action()
            if not indiv:
                x_s_all, x_b_all, l_all, x_s_active, x_b_active, l_active = self.followers_action()
                for i, follower in enumerate(self.all_followers):
                    if not follower.active:
                          continue
                    else:
                        x_s = follower.sell
                        x_b = follower.buy
                        l = follower.load

                        grad_s = p_s - np.multiply(self.eet_param.p_e, l + np.sum(l_all, axis=0)) - self.eet_param.p_soh * (
                                    x_s + np.sum(x_s_active, axis=0))
                        grad_b = -p_b + np.multiply(self.eet_param.p_e,
                                                    l + np.sum(l_all, axis=0)) - self.eet_param.p_soh * (
                                             x_b + np.sum(x_b_active, axis=0))
                        #print(f"Follower {i}, Action :{x_s}, {x_b}, {l}")
                        # Gradient Update
                        x_s_next = x_s + self.hyper_param.ve_step_size_follower * grad_s
                        x_b_next = x_b + self.hyper_param.ve_step_size_follower * grad_b
                        l_next = x_s_next - x_b_next + follower.usage
                        #print(f"Follower {i}, Action :{x_s_next}, {x_b_next}, {l_next}")
                        follower.update_direct([x_s_next, x_b_next])
                        #print(f"Follower {i}, Action :{follower.sell}, {follower.buy}, {follower.load}")
                #print(np.sum(l_all, axis=0))
                x_s_all_proj, x_b_all_proj, l_all_proj, x_s_active_proj, x_b_active_proj, l_active_proj = self.followers_action()

                #print(np.sum(l_all, axis=0))
                x_s_var = cp.Variable((self.eet_param.total_users, self.eet_param.time_horizon), nonneg=True)
                x_b_var = cp.Variable((self.eet_param.total_users, self.eet_param.time_horizon), nonneg=True)
                l_var = cp.Variable((self.eet_param.total_users, self.eet_param.time_horizon), nonneg=True)

                obj = cp.Minimize(cp.sum(cp.power(x_s_var - x_s_all_proj, 2) + cp.power(x_b_var - x_b_all_proj, 2) + cp.power(l_var - l_all_proj, 2)))

                constraints = []
                constraints += [cp.sum(x_s_var, axis=0) <= self.eet_param.c_s]
                constraints += [cp.sum(x_b_var, axis=0) <= self.eet_param.c_b]

                for i, follower in enumerate(self.all_followers):
                    if not follower.active:
                        constraints += [l_var[i] == follower.load]
                        constraints += [x_s_var[i] == follower.sell, x_b_var[i] == follower.buy]
                    else:
                        constraints += [l_var[i] == x_s_var[i] - x_b_var[i] + follower.usage]

                q_ess = self.eet_param.q_init
                for t in range(self.eet_param.time_horizon):
                    q_ess = self.eet_param.alpha * q_ess + self.eet_param.beta_s * (cp.sum(x_s_var[:, t])) - self.eet_param.beta_b * (cp.sum(x_b_var[:, t]))
                    constraints += [q_ess >= self.eet_param.q_min, q_ess <= self.eet_param.q_max]

                prob = cp.Problem(obj, constraints)
                result = prob.solve(solver='ECOS')

                diff = np.sum(np.power(x_s_var.value - x_s_all, 2) + np.power(x_b_var.value - x_b_all, 2) + np.power(l_var.value - l_all, 2))
                #print(x_s_var.value, x_s_all)
                for i, follower in enumerate(self.all_followers):
                    if not follower.active:
                        continue
                    else:
                        #print("XS : ", x_s_var.value[i], "\nXB", x_b_var.value[i], "\nL", l_var.value[i], "\n", follower.usage)
                        follower.update_direct([x_s_var.value[i], x_b_var.value[i]])
                        #print("LOAD :", follower.load)
            else:
                for i, follower in enumerate(self.all_followers):
                    if not follower.active:
                        continue
                    else:
                        x_s_all, x_b_all, l_all, x_s_active, x_b_active, l_active = self.followers_action()

                        x_s = follower.sell
                        x_b = follower.buy
                        l = follower.load

                        grad_s = p_s - np.multiply(self.eet_param.p_e, l + np.sum(l_all, axis=0)) - self.eet_param.p_soh*(x_s + np.sum(x_s_active, axis=0))
                        grad_b = -p_b + np.multiply(self.eet_param.p_e, l + np.sum(l_all, axis=0)) - self.eet_param.p_soh*(x_b + np.sum(x_b_active, axis=0))

                        # Gradient Update
                        x_s_next = x_s + self.hyper_param.ve_step_size_follower * grad_s
                        x_b_next = x_b + self.hyper_param.ve_step_size_follower * grad_b
                        l_next = x_s_next - x_b_next + follower.usage

                        #print(f"Follower {i}, Action :{x_s_next}, {x_b_next}, {l_next}")

                        # Projection
                        x_s_var = cp.Variable(self.eet_param.time_horizon, nonneg=True)
                        x_b_var = cp.Variable(self.eet_param.time_horizon, nonneg=True)
                        l_var = cp.Variable(self.eet_param.time_horizon, nonneg=True)

                        obj = cp.Minimize(cp.sum(cp.power(x_s_var - x_s_next, 2) + cp.power(x_b_var - x_b_next, 2) + cp.power(l_var - l_next, 2)))
                        constraints = [] #= [x_s_var >= np.zeros(self.eet_param.time_horizon), x_b_var >= np.zeros(self.eet_param.time_horizon), l_var >= np.zeros(self.eet_param.time_horizon)]
                        constraints += [x_s_var + np.sum(x_s_active, axis=0) - x_s <= self.eet_param.c_s]
                        constraints += [x_b_var + np.sum(x_b_active, axis=0) - x_b <= self.eet_param.c_b]
                        constraints += [l_var == x_s_var - x_b_var + follower.usage]
                        q_ess = self.eet_param.q_init
                        for t in range(self.eet_param.time_horizon):
                            q_ess = self.eet_param.alpha * q_ess + self.eet_param.beta_s*(x_s_var[t] -x_s[t] + np.sum(x_s_active[:, t])) - self.eet_param.beta_b*(x_b_var[t] - x_b[t] + np.sum(x_b_active[:, t]))
                            constraints += [q_ess >= self.eet_param.q_min, q_ess <= self.eet_param.q_max]

                        prob = cp.Problem(obj, constraints)
                        result = prob.solve(solver='ECOS')

                        diff += np.sum(np.power(x_s_var.value - x_s, 2) + np.power(x_b_var.value - x_b, 2) + np.power(l_var.value - l, 2))
                        follower.update_direct([x_s_var.value, x_b_var.value])
                        #print(f"Follower {i}, Action :{follower.sell}, {follower.buy}, {follower.load}")
            #load_curve += [np.sum(l_all, axis=0)]
            #print("x_s", x_s_active.reshape(-1))
            #print("x_b", x_b_active.reshape(-1))
            #print("l", l_active.reshape(-1))
            diff = np.sqrt(diff)
            #print(f'Iter {iter+1}, Difference of followers action : {diff} / Stopping criterion : {self.hyper_param.ve_eps}')
            if diff <= self.hyper_param.ve_eps:
                break
        print(
            f'Iter {iter + 1}, Difference of followers action : {diff} / Stopping criterion : {self.hyper_param.ve_eps}')
        #plt.figure()
        #for i, c in enumerate(load_curve):
        #    plt.plot(c, label=str(i+1))
        #plt.legend()
        #plt.show()
        x_s_all, x_b_all, l_all, x_s_active, x_b_active, l_active = self.followers_action()
        print("x_s", x_s_active.reshape(-1))
        print("x_b", x_b_active.reshape(-1))
        print("l", l_active.reshape(-1))
        return diff

    def is_ve(self):
        print("Check the followers VE")
        p_s, p_b = self.leader_action()
        for i, follower in enumerate(self.all_followers):
            if not follower.active:
                continue
            else:
                x_s_all, x_b_all, l_all, x_s_active, x_b_active, l_active = self.followers_action()

                x_s = follower.sell
                x_b = follower.buy
                l = follower.load

                grad_s = p_s - np.multiply(self.eet_param.p_e, l + np.sum(l_all, axis=0)) - self.eet_param.p_soh * (
                            x_s + np.sum(x_s_active, axis=0))
                grad_b = -p_b + np.multiply(self.eet_param.p_e, l + np.sum(l_all, axis=0)) - self.eet_param.p_soh * (
                            x_b + np.sum(x_b_active, axis=0))

                # Gradient Update
                x_s_next = x_s + self.hyper_param.ve_step_size_follower * grad_s
                x_b_next = x_b + self.hyper_param.ve_step_size_follower * grad_b
                l_next = x_s_next - x_b_next + follower.usage

                # print(f"Follower {i}, Action :{x_s_next}, {x_b_next}, {l_next}")

                # Projection
                x_s_var = cp.Variable(self.eet_param.time_horizon, nonneg=True)
                x_b_var = cp.Variable(self.eet_param.time_horizon, nonneg=True)
                l_var = cp.Variable(self.eet_param.time_horizon, nonneg=True)

                obj = cp.Minimize(cp.sum(
                    cp.power(x_s_var - x_s_next, 2) + cp.power(x_b_var - x_b_next, 2) + cp.power(l_var - l_next, 2)))
                constraints = []  # = [x_s_var >= np.zeros(self.eet_param.time_horizon), x_b_var >= np.zeros(self.eet_param.time_horizon), l_var >= np.zeros(self.eet_param.time_horizon)]
                constraints += [x_s_var + np.sum(x_s_active, axis=0) - x_s <= self.eet_param.c_s]
                constraints += [x_b_var + np.sum(x_b_active, axis=0) - x_b <= self.eet_param.c_b]
                constraints += [l_var == x_s_var - x_b_var + follower.usage]
                q_ess = self.eet_param.q_init
                for t in range(self.eet_param.time_horizon):
                    q_ess = self.eet_param.alpha * q_ess + self.eet_param.beta_s * (
                                x_s_var[t] - x_s[t] + np.sum(x_s_active[:, t])) - self.eet_param.beta_b * (
                                        x_b_var[t] - x_b[t] + np.sum(x_b_active[:, t]))
                    constraints += [q_ess >= self.eet_param.q_min, q_ess <= self.eet_param.q_max]

                prob = cp.Problem(obj, constraints)
                result = prob.solve(solver='ECOS')

                diff = np.sqrt(np.sum(
                    np.power(x_s_var.value - x_s, 2) + np.power(x_b_var.value - x_b, 2) + np.power(l_var.value - l, 2)))
                print(f"Follower {i}, Proximal Step Difference : {diff}")

    def followers_indiv_ve_iter(self, num_iter):
        print("Compute the followers VE")
        p_s, p_b = self.leader_action()
        diff = 0

        load_curve = []

        for iter in range(num_iter):
            diff = 0

            x_s_all, x_b_all, l_all, x_s_active, x_b_active, l_active = self.followers_action()

            for follower in self.all_followers:
                if not follower.active:
                    continue
                else:
                    x_s_all, x_b_all, l_all, x_s_active, x_b_active, l_active = self.followers_action()

                    x_s = follower.sell
                    x_b = follower.buy
                    l = follower.load

                    grad_s = p_s - np.multiply(self.eet_param.p_e, l + np.sum(l_all, axis=0)) - self.eet_param.p_soh*(x_s + np.sum(x_s_active, axis=0))
                    grad_b = -p_b + np.multiply(self.eet_param.p_e, l + np.sum(l_all, axis=0)) - self.eet_param.p_soh*(x_b + np.sum(x_b_active, axis=0))

                    # Gradient Update
                    x_s_next = x_s + self.hyper_param.ve_step_size_follower * grad_s
                    x_b_next = x_b + self.hyper_param.ve_step_size_follower * grad_b
                    l_next = x_s_next - x_b_next + follower.usage

                    # Projection
                    x_s_var = cp.Variable(self.eet_param.time_horizon, nonneg=True)
                    x_b_var = cp.Variable(self.eet_param.time_horizon, nonneg=True)
                    l_var = cp.Variable(self.eet_param.time_horizon, nonneg=True)

                    obj = cp.Minimize(cp.sum(cp.power(x_s_var - x_s_next, 2) + cp.power(x_b_var - x_b_next, 2) + cp.power(l_var - l_next, 2)))
                    constraints = [] #= [x_s_var >= np.zeros(self.eet_param.time_horizon), x_b_var >= np.zeros(self.eet_param.time_horizon), l_var >= np.zeros(self.eet_param.time_horizon)]
                    constraints += [x_s_var <= self.eet_param.c_s/self.eet_param.active_users]
                    constraints += [x_b_var <= self.eet_param.c_b/self.eet_param.active_users]
                    constraints += [l_var == x_s_var - x_b_var + follower.usage]
                    q_ess = 0. # self.eet_param.q_init
                    for t in range(self.eet_param.time_horizon):
                        q_ess = self.eet_param.alpha * q_ess + self.eet_param.beta_s*(x_s_var[t]) - self.eet_param.beta_b*(x_b_var[t])
                        constraints += [q_ess >= self.eet_param.q_min/self.eet_param.active_users, q_ess <= self.eet_param.q_max/self.eet_param.active_users]

                    prob = cp.Problem(obj, constraints)
                    result = prob.solve(solver='ECOS')

                    diff += np.sqrt(np.sum(np.power(x_s_var.value - x_s, 2) + np.power(x_b_var.value - x_b, 2) + np.power(l_var.value - l, 2)))
                    follower.update_direct([x_s_var.value, x_b_var.value])

            #x_s_all, x_b_all, l_all, x_s_active, x_b_active, l_active = self.followers_action()
            #load_curve += [np.sum(l_all, axis=0)]
            print(f'Iter {iter+1}, Difference of followers action : {diff} / Stopping criterion : {self.hyper_param.ve_eps}')
            if diff <= self.hyper_param.ve_eps:
                break
        #plt.figure()
        #for i, c in enumerate(load_curve):
        #    plt.plot(c, label=str(i+1))
        #plt.legend()
        #plt.show()
        return diff

    def inequality_const_value(self):
        x_s_all, x_b_all, l_all, x_s_active, x_b_active, l_active = self.followers_action()

        c_1 = -x_s_active.T # -x_s
        c_2 = -x_b_active.T # -x_b
        c_3 = -l_active.T # -l

        q = np.zeros(self.eet_param.time_horizon)
        q[0] = self.eet_param.q_init

        for t in range(1, self.eet_param.time_horizon):
            q[t] = self.eet_param.alpha*q[t-1] + self.eet_param.beta_s * np.sum(x_s_active[:, t]) - self.eet_param.beta_b * np.sum(x_b_active[:, t])

        c_4 = -q # -q
        c_5 = q - self.eet_param.q_max * np.ones(self.eet_param.time_horizon) # q - q_max

        c_6 = np.sum(x_s_active, axis=0) - self.eet_param.c_s # sum(x_s) - c_s
        c_7 = np.sum(x_b_active, axis=0) - self.eet_param.c_b # sum(x_b) - c_b

        return [c_1, c_2, c_3, c_4, c_5, c_6, c_7]

    def is_active_constraints(self):
        [c_1, c_2, c_3, c_4, c_5, c_6, c_7] = self.inequality_const_value()
        f = lambda x: np.abs(x) <= self.hyper_param.active_epsilon
        active = np.hstack((f(c_1).reshape(-1), f(c_2).reshape(-1), f(c_3).reshape(-1), f(c_4).reshape(-1), f(c_5).reshape(-1), f(c_6).reshape(-1), f(c_7).reshape(-1)))

        return active

    def gradient_follower_wrt_leader(self):
        print("Computing Gradient")
        x_s_all, x_b_all, l_all, x_s_active, x_b_active, l_active = self.followers_action()

        n = self.eet_param.active_users
        t = self.eet_param.time_horizon
        nc = (5*n+4)*t

        a = self.eet_param.alpha
        bb = self.eet_param.beta_b
        bs = self.eet_param.beta_s
        pe = self.eet_param.p_e
        psoh = self.eet_param.p_soh
        ptax = self.eet_param.p_tax

        MF = np.zeros((nc, n*t*2))

        MF[:n*t, :n*t] = np.kron(np.diag(pe+psoh*np.ones(t)), np.eye(n)+np.ones(n))
        MF[:n*t, n*t:] = np.kron(np.diag(-pe), np.eye(n)+np.ones(n))
        MF[n*t:2*n * t, :n * t] = np.kron(np.diag(-pe), np.eye(n) + np.ones(n))
        MF[n*t:2*n * t, n * t:] = np.kron(np.diag(pe + psoh * np.ones(t)), np.eye(n) + np.ones(n))
        MF[2*n*t:5*n*t] = np.kron(np.array([[-1, 0], [0, -1], [-1, 1]]), np.eye(n*t))

        qs = np.zeros((t, n*t))
        qb = np.zeros((t, n*t))

        qs[0, :n] = bs*np.ones(n)
        qb[0, :n] = bb*np.ones(n)
        for i in range(1, t):
            qs[i] = a*qs[i-1]
            qs[i, n*i:n*(i+1)] = bs*np.ones(n)
            qb[i] = a*qb[i-1]
            qb[i, n*i:n*(i+1)] = bs*np.ones(n)

        MF[5*n*t:5*n*t+t, :n*t] = -qs
        MF[5*n*t:5*n*t+t, n*t:] = qb
        MF[5*n*t+t:5*n*t+2*t, :n*t] = qs
        MF[5*n*t+t:5*n*t+2*t, n*t:] = -qb
        MF[5*n*t+2*t:, :] = np.kron(np.eye(2*t), np.ones(n))

        MLF = -np.kron(np.kron(np.array([[1, 0], [0, -1]]), np.eye(t)), np.ones((n, 1)))

        ML = np.zeros((nc, t*2))
        ML[:2*n*t] = np.kron(np.kron(np.array([[-1, 0], [0, 1]]), np.eye(t)), np.ones((n, 1)))

        MFF = np.zeros((n*t*2, n*t*2))
        MFF[:n*t, :n*t] = np.kron(np.diag(pe+psoh*np.ones(t)), 2*(np.eye(n)+np.ones((n,n))))
        MFF[n * t:, n * t:] = np.kron(np.diag(pe+psoh*np.ones(t)), 2*(np.eye(n)+np.ones((n,n))))
        MFF[:n*t, n*t:] = np.kron(np.diag(-pe), 2*(np.eye(n)+np.ones((n,n))))
        MFF[n * t:, :n * t] = np.kron(np.diag(-pe), 2 * (np.eye(n) + np.ones((n, n))))

        active = self.is_active_constraints()
        MF = np.vstack([MF[:2*n*t], MF[2*n*t:][active]])
        ML = np.vstack([ML[:2*n*t], ML[2*n*t:][active]])

        MFF_inv = np.linalg.pinv(MFF)
        dw_dy = MFF_inv@MF.T@np.linalg.pinv(MF@MFF_inv@MF.T)@(MF@MFF_inv@MLF-ML)-MFF_inv@MLF

        return dw_dy


    def grad_one_iteration(self):
        s = time.time()
        x = self.compute_followers_ve()
        e = time.time()
        a = e-s
        print("VE TIME :", e-s)
        self.update_followers(x)
        #print(len(self.grad_history.leader_decision_history))
        #self.update_grad_history(self.leader_action(), self.leader_utility(), self.followers_action(),
        #                         self.followers_utility())
        #print(len(self.grad_history.leader_decision_history))
        p_prev = self.leader_action()
        s2 = time.time()
        grad = self.compute_leader_gradient()
        e2 = time.time()
        b = e2-s2
        print("Grad TIME :", e2-s2)
        if len(self.grad_history.time_history) == 0:
            self.grad_history.update(self.leader_action(), self.leader_utility(), self.followers_action(), self.followers_utility(), a+b)
        else:
            self.grad_history.update(self.leader_action(), self.leader_utility(), self.followers_action(), self.followers_utility(), a + b - self.grad_history.time_history[-1])
        self.save_data()
        self.leader.update_grad(grad, self.grad_step_size)
        diff1 = np.sqrt(np.sum(np.power(self.leader_action() - p_prev, 2)))
        diff2 = np.sqrt(np.sum(np.power(grad, 2)))
        print("Sum Time :", a+b)
        return diff1, diff2

    def grad_iterations(self):
        if self.grad_history.updated_cnt:
            self.leader.update_direct(self.grad_history.leader_decision_history[-1])
            for i in range(self.evs):
                self.update_followers(self.grad_history.followers_decision_history[-1])
        else:
            self.initialize_action()

        for i in range(self.grad_max_iter):
            print("Already", self.grad_history.updated_cnt, "iteration progressed")
            print("Grad Iteration :", i+1)
            self.print_information()
            diff1, diff2 = self.grad_one_iteration()

            if (np.abs(diff1 < self.grad_eps) or np.abs(diff2 < self.grad_eps))and i > 10:
                #self.save_data()
                print("Grad Iteration Over")
                print("grad diff :", np.abs(diff1), "action diff :", np.abs(diff2), "smaller than eps :", self.grad_eps)
                break
            #if i % 1 == 0:
            #    print("diff :", diff)
            #    self.save_data()
        print("Grad Maximum Iteration Over")
        self.save_data()
        return 0

    def compute_leader_gradient(self):
        x_s_all, x_b_all, l_all, x_s_active, x_b_active, l_active = self.followers_action()

        t = self.eet_param.time_horizon
        n = self.eet_param.active_users
        l = np.sum(l_all, axis=0)
        ptax = self.eet_param.p_tax
        pe = self.eet_param.p_e

        grad_ULL = -2*ptax*np.hstack([self.leader.sell, self.leader.buy])
        grad_ULF = np.kron(np.array([-2, 2]), np.kron(np.multiply(pe, l), np.ones(n)))
        grad_FL = self.gradient_follower_wrt_leader()

        #print(grad_ULL.shape, grad_ULF.shape, grad_FL.shape)
        grad = grad_ULL + grad_ULF@grad_FL

        return grad[:t], grad[t:]

    def se_algorithm(self, indiv=False):
        diff = 0
        par, ec, l_util, f_utils = self.info_func()

        par_hist = [par]
        ec_hist = [ec]
        l_util_hist = [l_util]

        for iter in range(self.hyper_param.grad_max_iter):
            diff_f = self.followers_ve_iter(num_iter=self.hyper_param.ve_max_iter, indiv=indiv)
            par, ec, l_util, f_utils = self.info_func()
            grad_s, grad_b = self.compute_leader_gradient()

            diff_l = self.leader.update_grad([grad_s, grad_b], self.hyper_param.grad_step_size)
            diff = np.sqrt(diff_f**2 + diff_l**2)

            par_hist += [par]
            ec_hist += [ec]
            l_util_hist += [l_util]
            print(f'Iter {iter+1}, Difference of follower and leader action : {diff} / Stopping criterion : {self.hyper_param.ve_eps}')
            #print(self.leader.sell, self.leader.buy)
            print(f'SE PAR History {par_hist}')
            if diff <= self.hyper_param.ve_eps:
                break

        return diff, par_hist, ec_hist, l_util_hist

    def spne_algorithm(self, fix_leader=True):
        diff = 0
        par, ec, l_util, f_utils = self.info_func()

        par_hist = [par]
        ec_hist = [ec]
        l_util_hist = [l_util]

        for iter in range(self.hyper_param.spne_max_iter):
            if fix_leader:
                self.leader.update_direct([np.zeros(self.eet_param.time_horizon), np.zeros(self.eet_param.time_horizon)])
                diff_l = 0
            else:
                p_s = self.leader.sell
                p_b = self.leader.buy
                ratio = (1-2*self.hyper_param.ve_step_size_leader*self.eet_param.p_tax)
                diff_l = ratio * np.linalg.norm(np.concatenate([p_s, p_b]))
                self.leader.update_direct([ratio*p_s, ratio*p_b])

            diff_f = self.followers_ve_iter(num_iter=1, indiv=True)

            diff = np.sqrt(diff_l**2 + diff_f**2)

            par, ec, l_util, f_utils = self.info_func()

            par_hist += [par]
            ec_hist += [ec]
            l_util_hist += [l_util]
            print(f'Iter {iter+1}, Difference of follower and leader action : {diff} / Stopping criterion : {self.hyper_param.ve_eps}')
            print(f'SPNE PAR History {par_hist}')
            if diff <= self.hyper_param.ve_eps:
                break

        return diff, par_hist, ec_hist, l_util_hist

    def indiv_spne_algorithm(self, fix_leader=True):
        diff = 0
        par_hist = []
        ec_hist = []
        for iter in range(self.hyper_param.spne_max_iter):
            if fix_leader:
                self.leader.update_direct([np.zeros(self.eet_param.time_horizon), np.zeros(self.eet_param.time_horizon)])
                diff_l = 0
            else:
                p_s = self.leader.sell
                p_b = self.leader.buy
                ratio = (1-2*self.hyper_param.ve_step_size_leader*self.eet_param.p_tax)
                diff_l = ratio * np.linalg.norm(np.concatenate([p_s, p_b]))
                self.leader.update_direct([ratio*p_s, ratio*p_b])

            diff_f = self.followers_indiv_ve_iter(num_iter=1)

            diff = np.sqrt(diff_l**2 + diff_f**2)

            par, ec, l_util, f_utils = self.info_func()

            par_hist += [par]
            ec_hist += [ec]
            print(f'Iter {iter+1}, Difference of follower and leader action : {diff} / Stopping criterion : {self.hyper_param.ve_eps}')
            print(f'PAR History {par_hist}')
            if diff <= self.hyper_param.ve_eps:
                break

        return diff, par_hist, ec_hist


if __name__=="__main__":
    load = np.load("./data/load_123.npy", allow_pickle=True)[:, :24]
    pv = np.load("./data/E_PV.npy", allow_pickle=True)[:24]

    par_se_list = []
    par_spne_list = []
    par_indiv_list = []
    ec_se_list = []
    ec_spne_list = []
    ec_indiv_list = []
    #np.save("par_data.npy", [par_se_list, par_spne_list])
    #np.save("./result/par_data_indiv.npy", par_indiv_list)#[par_se_list, par_spne_list])
    for i in range(1, 100):
        eet_input = {'total_users': 100,
                     'active_users': i,
                     'time_horizon': 24, }
        game_se = EETGame(load, pv, eet_input=eet_input)
        game_spne = EETGame(load, pv, eet_input=eet_input)
        #game_indiv = EETGame(load, pv, eet_input=eet_input)
        #game_se.followers_ve_iter(num_iter=5)
        #print(game_se.compute_leader_gradient())

        _, par_se, ec_se = game_se.se_algorithm()
        _, par_spne, ec_spne = game_spne.spne_algorithm(fix_leader=False) # 2.5278279782256923
        #_, par_indiv, ec_indiv = game_indiv.indiv_spne_algorithm(fix_leader=False)

        plt.figure()
        plt.plot(par_se+[par_se[-1] for _ in range(len(par_spne)-len(par_se))], color='r', label='Stackelberg Equilibrium')
        plt.plot(par_spne, color='k', label='Subgame Perfect Nash Equilibrium')
        plt.legend()
        #plt.savefig(f'./result/par_user_{str(i)}')
        plt.show(block=False)
        plt.pause(5)
        plt.close()

        plt.figure()
        plt.plot(ec_se + [ec_se[-1] for _ in range(len(par_spne) - len(par_se))], color='r',
                 label='Stackelberg Equilibrium')
        plt.plot(ec_spne, color='k', label='Subgame Perfect Nash Equilibrium')
        plt.legend()
        #plt.savefig(f'./result/ec_user_{str(i)}')
        plt.show(block=False)
        plt.pause(5)
        plt.close()

        #plt.figure()
        #plt.plot(par_indiv, color='k', label='Subgame Perfect Nash Equilibrium(Individiual)')
        #plt.legend()
        #plt.savefig(f'./result/user_{str(i)}_indiv')
        #plt.show(block=False)
        #plt.pause(5)
        #plt.close()

        par_se_list += [min(par_se)]
        par_spne_list += [par_spne[-1]]

        ec_se_list += [min(ec_se)]
        ec_spne_list += [ec_spne[-1]]
        #par_indiv_list += [par_indiv[-1]]


        print("SE PAR :", [ '%.4f' % elem for elem in par_se_list])
        print("SPNE PAR :", ['%.4f' % elem for elem in par_spne_list])
        print("SE EC :", ['%.4f' % elem for elem in ec_se_list])
        print("SPNE EC :", ['%.4f' % elem for elem in ec_spne_list])

        #print("Individual SPNE PAR :", ['%.4f' % elem for elem in par_spne_list])
    np.save("./result/par_data.npy", [par_se_list, par_spne_list])
    np.save("./result/ec_data.npy", [ec_se_list, ec_spne_list])
    #np.save("./result/par_data_indiv.npy", par_indiv_list)

    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(par_se_list, color='r', label='Stackelberg Equilibrium')
    plt.plot(par_spne_list, color='k', label='Subgame Perfect Nash Equilibrium')

    # plt.plot(z, label='Subgame Perfect Nash Equilibrium (Individual)')
    plt.xlabel("Number of Active Users", fontsize=20)
    plt.ylabel("PAR", fontsize=20)
    plt.xticks([0, 5, 10, 15, 20], fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    #plt.savefig(f'./result/1_active_users', bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(ec_se_list, color='r', label='Stackelberg Equilibrium')
    plt.plot(ec_spne_list, color='k', label='Subgame Perfect Nash Equilibrium')

    # plt.plot(z, label='Subgame Perfect Nash Equilibrium (Individual)')
    plt.xlabel("Number of Active Users", fontsize=20)
    plt.ylabel("EC", fontsize=20)
    plt.xticks([0, 5, 10, 15, 20], fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    #plt.savefig(f'./result/1_active_users', bbox_inches='tight')
    plt.show()