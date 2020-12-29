import os
import pandas as pd
import random
import numpy as np

name = os.path.join('./../datasets', 'sidewalk_psi_accy.txt')

df = pd.read_csv(name, sep=" ", engine="python", encoding="ISO-8859-1", header=None)

begin = 2480
end = 2529
yaw_r = [0, -25] # -2, 2
y_r = [0, 1.2] # -0.1, 0.1
cres_decres = True # False True

yaw_factor = (yaw_r[1] - yaw_r[0])/((end - begin) + 2)
yaw_l = np.arange(yaw_r[0], yaw_r[1], yaw_factor).tolist()

y_factor = (y_r[1] - y_r[0])/((end - begin) + 2)
y_l = np.arange(y_r[0], y_r[1], y_factor).tolist()

c = 0

for i in range(begin, end + 1):
    if cres_decres:
        df.at[i, 3] = float("%.4f" % random.uniform(yaw_l[c], yaw_l[c + 1]))
        df.at[i, 4] = float("%.4f" % random.uniform(y_l[c], y_l[c + 1]))
        c += 1
    else:
        df.at[i, 3] = float("%.4f" % random.uniform(yaw_r[0], yaw_r[1])) #desvio em yaw -1, 1
        df.at[i, 4] = float("%.4f" % random.uniform(y_r[0], y_r[1])) #desvio em y -0.08, 0.08

print('Finalizou', begin, end)

df.to_csv(name, sep=" ", index=False, header=None)

