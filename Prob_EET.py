import numpy as np
import cvxpy as cp
import pickle
import matplotlib.pyplot as plt
import time


class HyperParameters:
    def __init__(self, hyper_input=None):
        self.grad_step_size = 0.005
        self.grad_max_iter = 50
        self.grad_eps = 1e-4

        self.ve_step_size_leader = 0.3
        self.ve_step_size_follower = .05
        self.ve_max_iter = 10

        self.ve_eps = 1e-5
        self.spne_max_iter = 50
        self.active_epsilon = 1e-5

        self.prox_gamma = 1
        self.prox_eps = 1e-5
        self.prox_max_iter = 1000


class EETParameters:
    def __init__(self, eet_input=None):
        self.total_users = 100
        self.active_users = 10
        self.passive_users = self.total_users - self.active_users
        self.time_horizon = 12
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
    def __init__(self, initial_decision=None, time_horizon=12):
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
    def __init__(self, energy_use, is_active=True, time_horizon=12):
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

        self.leader = Leader()
        self.active_followers = []
        self.passive_followers = []
        self.usage_matrix = e_ha_matrix - e_pv_matrix
        for i in range(self.eet_param.active_users):
            self.active_followers += [Follower(energy_use = self.usage_matrix[i], is_active=True)]
        for i in range(self.eet_param.active_users, self.eet_param.total_users):
            self.passive_followers += [Follower(energy_use = self.usage_matrix[i], is_active=False)]

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
        ec = -np.sum(np.multiply(self.eet_param.p_e, np.power(load, 2)))
        l_util = -np.sum(np.multiply(self.eet_param.p_e, np.power(load, 2)))-self.eet_param.p_tax*np.sum(np.power(p_s, 2) + np.power(p_b, 2))
        f_util_list = []
        for i in range(self.eet_param.total_users):
            f_util_list += [np.sum(np.multiply(p_s, x_s_all[i]) - np.multiply(p_b, x_b_all[i]))
                            - np.sum(np.multiply(self.eet_param.p_e, np.multiply(l_all[i], np.sum(l_all, axis=0))))
                            - self.eet_param.p_soh*(np.sum(np.multiply(x_s_all[i], np.sum(x_s_all, axis=0))
                                                           + np.multiply(x_b_all[i], np.sum(x_b_all, axis=0))))]

        return par, ec, l_util, f_util_list

    def followers_ve_iter(self, num_iter):
        print("Compute the followers VE")
        p_s, p_b = self.leader_action()
        diff = 0

        for iter in range(num_iter):
            diff = 0
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
                    constraints += [x_s_var + np.sum(x_s_active, axis=0) - x_s <= self.eet_param.c_s]
                    constraints += [x_b_var + np.sum(x_b_active, axis=0) - x_b <= self.eet_param.c_b]
                    constraints += [l_var == x_s_var - x_b_var + follower.usage]
                    q_ess = self.eet_param.q_init
                    for t in range(self.eet_param.time_horizon):
                        q_ess = self.eet_param.alpha * q_ess + self.eet_param.beta_s*(x_s_var[t] -x_s[t] + np.sum(x_s_active[:, t])) - self.eet_param.beta_b*(x_b_var[t] - x_b[t] + np.sum(x_b_active[:, t]))
                        constraints += [q_ess >= self.eet_param.q_min, q_ess <= self.eet_param.q_max]

                    prob = cp.Problem(obj, constraints)
                    result = prob.solve(solver='ECOS')

                    diff += np.sqrt(np.sum(np.power(x_s_var.value - x_s, 2) + np.power(x_b_var.value - x_b, 2) + np.power(l_var.value - l, 2)))
                    follower.update_direct([x_s_var.value, x_b_var.value])

            print(f'Iter {iter+1}, Difference of followers action : {diff} / Stopping criterion : {self.hyper_param.ve_eps}')
            if diff <= self.hyper_param.ve_eps:
                break
        return diff

    def inequality_const_value(self):
        dest = self.followers_action()
        x_nm = - dest
        x_m = np.zeros((2, self.eet_param.time_horizon))
        x_nm_mu = np.zeros((self.eet_param.time_horizon, self.eet_param.time_horizon))
        x_m_mu = np.zeros((2, self.eet_param.time_horizon))

        x_m[0, :] = np.sum(np.multiply(dest, np.kron(np.ones(self.eet_param.time_horizon), self.eet_param.time_horizon.reshape(-1, 1))), axis=0) - self.eet_param.time_horizon
        x_m[1, :] = np.sum(dest, axis=0) - self.eet_param.time_horizon

        z_nm = np.copy(x_nm)
        z_m = np.copy(x_m)

        f = lambda x: np.abs(x) <= self.active_epsilon
        x_nm_mu += f(x_nm)
        x_m_mu += f(x_m)
        return [x_nm, x_m, z_nm, z_m, x_nm_mu, x_m_mu]

    def is_active_constraints(self):
        [x_nm, x_m, z_nm, z_m, x_nm_mu, x_m_mu] = self.inequality_const_value()
        f = lambda x: np.abs(x) <= self.active_epsilon
        active = np.hstack((f(x_nm).reshape(-1), f(x_m).reshape(-1), f(z_nm).reshape(-1), f(z_m).reshape(-1), f(x_nm_mu).reshape(-1), f(x_m_mu).reshape(-1)))
        return active

    def compute_leader_gradient_direct(self):
        print("Computing Gradient")
        dest = self.followers_action()
        Dxh = np.zeros((2*self.eet_param.active_users*self.eet_param.active_users+2*self.eet_param.active_users+2*self.eet_param.active_users, self.eet_param.active_users))
        for i in range(self.eet_param.active_users):
            Dxh[i*self.eet_param.active_users:(i+1)*self.eet_param.active_users] = self.eet_param.active_users[i, 1]*self.eet_param.active_users[i]*np.eye(self.eet_param.active_users)

        Dyh = np.zeros((2*self.eet_param.active_users*self.eet_param.active_users+2*self.eet_param.active_users+2*self.eet_param.active_users, 3*self.eet_param.active_users*self.eet_param.active_users+2*self.eet_param.active_users+self.eet_param.active_users))
        m = self.eet_param.active_users
        n = self.eet_param.time_horizon
        Dyh[:m*n, :m*n] = np.kron(np.diag(self.followers[:, 2])+np.kron(np.ones((n, 1)), self.followers[:, 2]), np.eye(m))
        Dyh[:m*n, 2*m*n:3*m*n] = -np.eye(m*n)
        Dyh[:m * n, 3 * m * n:3 * m * n + m] = np.kron(self.evs_load.reshape(-1, 1), np.eye(m))
        Dyh[:m * n, 3 * m * n + m:3 * m * n + 2* m] = np.kron(np.ones((n, 1)), np.eye(m))
        Dyh[:m * n, 3 * m * n + 2 * m:3 * m * n + 2 * m + n] = np.kron(np.eye(n), np.ones((m, 1)))

        Dyh[m*n : 2* m * n+2*m, 2 * m * n:3*m*n+2*m] = np.diag(np.hstack((dest.reshape(-1), np.sum(np.multiply(dest, np.kron(np.ones(self.eet_param.active_users), self.eet_param.active_users.reshape(-1, 1))), axis=0) - self.max_elec,
                                                                          np.sum(dest, axis=0) - self.eet_param.active_users)))
        Dyh[2*m*n+2*m:2*m*n+2*m+n, :m*n] = np.kron(np.eye(n), np.ones(m))

        Dxg = np.zeros((3*m*(n+2), m))
        Dyg = np.zeros((3*m*(n+2), 3*m*n+2*m+n))
        Dyg[:m*n, :m*n] = -np.eye(m*n)
        Dyg[m*n:m*n+2*m, :m*n] = np.kron(np.vstack((self.eet_param.active_users, np.ones(n))), np.eye(m))
        #Dyg[m*n+2*m:2*(m*n+2*m), m*n:2*m*n] = Dyg[:m*n+2*m, :m*n]
        Dyg[2*(m*n+2*m):,2*m*n:2*m*n+m*n+2*m] = -np.eye(m*n+2*m)

        Dxh_wave = Dxh
        Dyh_wave = Dyh

        #print(Dyh_wave.shape, Dxh_wave.shape)

        s = time.time()
        eps = 1e-5
        Dy_var = cp.Variable((3*m*n+2*m+n, m))
        obj = cp.Minimize(1)
        const = [Dyh_wave@Dy_var + Dxh_wave <= eps]
        const = [Dyh_wave @ Dy_var + Dxh_wave >= -eps]
        prob = cp.Problem(obj, const)
        result = prob.solve(solver='ECOS')
        #print(prob.status)
        dy = Dy_var.value
        e = time.time()

        #print("Gradient Computing time :", e-s)
        v = np.sum(dest, axis=0) - self.eet_param.active_users

        dxj = np.zeros(m)
        dyj = 2 * np.kron(np.ones(n), v)
        dj = dxj - dyj@dy[:m*n, :]
        #print(dy)

        print("DJ :", dj)
        return dj

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
        g_s_1 = -2*self.eet_param.p_tax*self.leader.sell
        g_b_1 = -2*self.eet_param.p_tax*self.leader.buy

        g_s_2 = -2*np.multiply(self.eet_param.p_e, np.sum(l_all, axis=0))
        g_b_2 = 2*np.multiply(self.eet_param.p_e, np.sum(l_all, axis=0))

        T = self.eet_param.time_horizon

        H = np.zeros((4 * T, 4 * T))

        H[:2 * T, : 2 * T] = -2 * np.ones((2 * T, 2 * T)) - (self.eet_param.active_users + 3) * np.eye(2 * T)
        H[2 * T:, : 2 * T] = (self.eet_param.active_users + 1) * np.eye(2 * T)
        H[:2 * T, 2 * T:] = (self.eet_param.active_users + 1) * np.eye(2 * T)

        B = np.kron(np.array([1, 0, 0, -1, 1, 0, 0, 1]).reshape(4, 2), np.eye(T))

        x = np.concatenate([g_s_2, g_b_2])@((-np.linalg.inv(H)@B)[:2 * T, :])

        grad_s = g_s_1 + x[:T]
        grad_b = g_b_1 + x[T:]

        #print(g_s_1, x[:T])
        #print(g_b_1, x[T:])

        #grad_s = g_s_1 + g_s_2 @ g_s_3
        #grad_b = g_b_1 + g_b_2 @ g_b_3
        return grad_s, grad_b

    def se_algorithm(self):
        diff = 0
        par_hist = []
        for iter in range(self.hyper_param.grad_max_iter):
            diff_f = self.followers_ve_iter(num_iter=self.hyper_param.ve_max_iter)
            grad_s, grad_b = self.compute_leader_gradient()
            diff_l = self.leader.update_grad([grad_s, grad_b], self.hyper_param.grad_step_size)
            diff = np.sqrt(diff_f**2 + diff_l**2)

            par, ec, l_util, f_utils = self.info_func()

            par_hist += [par]

            print(f'Iter {iter+1}, Difference of follower and leader action : {diff} / Stopping criterion : {self.hyper_param.ve_eps}')
            #print(self.leader.sell, self.leader.buy)
            print(f'PAR History {par_hist}')
            if diff <= self.hyper_param.ve_eps:
                break

        return diff, par_hist

    def spne_algorithm(self, fix_leader=True):
        diff = 0
        par_hist = []
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

            diff_f = self.followers_ve_iter(num_iter=1)

            diff = np.sqrt(diff_l**2 + diff_f**2)

            par, ec, l_util, f_utils = self.info_func()

            par_hist += [par]
            print(f'Iter {iter+1}, Difference of follower and leader action : {diff} / Stopping criterion : {self.hyper_param.ve_eps}')
            print(f'PAR History {par_hist}')
            if diff <= self.hyper_param.ve_eps:
                break

        return diff, par_hist


if __name__=="__main__":
    load = np.load("./data/load_123.npy", allow_pickle=True)[:, 6:18]
    pv = np.load("./data/E_PV.npy", allow_pickle=True)[6:18]
    game_se = EETGame(load, pv)
    game_spne = EETGame(load, pv)
    #game.compute_leader_gradient()
    _, par_se = game_se.se_algorithm()
    _, par_spne = game_spne.spne_algorithm(fix_leader=False) # 2.5278279782256923

    plt.figure()
    plt.plot(par_se, color='r', label='Stackelberg Equilibrium')
    plt.plot(par_spne, color='k', label='Subgame Perfect Nash Equilibrium')
    plt.legend()
    plt.show()

