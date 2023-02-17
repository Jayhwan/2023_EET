import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import truncnorm


load1 = np.array([0.64, 1.12, 1.32, 1.3, 1.13, 0.9,
                  0.73, 0.42, 0.30, 0.33, 0.41, 0.51])
load2 = np.array([0.35, 0.31, 0.39, 0.59, 0.78, 0.85,
                  1.01, 1.27, 1.33, 1.23, 0.93, 0.65])
load3 = np.array([0.5, 0.43, 0.42, 0.36, 0.38, 0.58,
                  0.69, 0.91, 0.95, 0.88, 0.71, 0.54])
load1 = 1*12*load1/np.sum(load1)
load2 = 1*12*load2/np.sum(load2)
load3 = 0.5*12*load3/np.sum(load3)
plt.plot(load1)
plt.plot(load2)
plt.plot(load3)
plt.show()
loads = np.array([load1, load2, load3])
a=0.1
load = np.array([loads[i%3] + truncnorm.rvs(-a, a, size=12) for i in range(100)])
plt.plot(load)
plt.show()
#np.save("three_type_load.npy", load)
load = np.load("./data/load_123.npy", allow_pickle=True)
pv = np.load("./data/E_PV.npy", allow_pickle=True)

load = load[:, :24].T
pv = pv[:24]
plt.plot(load)
plt.plot(pv)
plt.show()