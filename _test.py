import numpy as np
import matplotlib.pyplot as plt

a = np.array([9.36178665e-03, 6.87023124e-10, 3.12902311e-10, 8.26743390e-02, 1.00426572e-01, 5.82654712e-10])
b = np.array([3.16381796e-10, 6.09806593e-10, 1.87426294e-01, 3.15205737e-10, 3.15224108e-10, 7.15984898e-10])
c = np.array([0.51476364, 0.61560001, 0.52108892, 0.51266464, 0.51494659, 0.42466313])

d = np.array([9.30175149e-03, 6.88591196e-10, 3.13657306e-10, 8.27350695e-02, 1.00433820e-01, 5.84089728e-10])
e = np.array([3.17074782e-10, 6.11284795e-10, 1.87434043e-01, 3.16001932e-10, 3.15981764e-10, 7.17586652e-10])
f = np.array([0.51470361, 0.61560001, 0.52108117, 0.51272537, 0.51495384, 0.42466313])

print(np.sum(np.sqrt(np.power(a-d, 2)+np.power(b-e, 2)+np.power(c-f, 2))))

"""[x, y] = np.load("./result/par_data.npy", allow_pickle=True)
z = np.load("./result/par_data_indiv.npy", allow_pickle=True)

plt.figure(figsize=(14, 6), dpi=300)
plt.plot(y, color='k', label='Subgame Perfect Nash Equilibrium')
plt.plot(x, color='r', label='Stackelberg Equilibrium')

#plt.plot(z, label='Subgame Perfect Nash Equilibrium (Individual)')
plt.xlabel("Number of Active Users", fontsize=20)
plt.ylabel("PAR", fontsize=20)
plt.xticks([0, 5, 10, 15, 20], fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.savefig(f'./result/2_active_users', bbox_inches='tight')
plt.show()
print(x, y)


[ec_se_list, ec_spne_list] = np.load("./result/ec_data.npy", allow_pickle=True)
plt.figure(figsize=(14, 6), dpi=300)
plt.plot(ec_spne_list, color='k', label='Subgame Perfect Nash Equilibrium')
plt.plot(ec_se_list, color='r', label='Stackelberg Equilibrium')

# plt.plot(z, label='Subgame Perfect Nash Equilibrium (Individual)')
plt.xlabel("Number of Active Users", fontsize=20)
plt.ylabel("EC", fontsize=20)
plt.xticks([0, 5, 10, 15, 20], fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.savefig(f'./result/2_active_users_ec', bbox_inches='tight')
plt.show()"""