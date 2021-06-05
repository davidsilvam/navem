import os
import pandas as pd
import random
import numpy as np

name = os.path.join('./../../datasets', 'indoor_dataset_velx_vely.txt')

df = pd.read_csv(name, sep=" ", engine="python", encoding="ISO-8859-1", header=None, names=['img_dataset', 'img_original', 'folder', 'x', 'y'])

begin = 0
end = 19
vel_variation = {}
vel_variation['x'] = [0, 3] # -2, 200 min max
vel_variation['y'] = [-0.08, 0.08] # -0.1, 0.1 min max

# =======   x     y   ====
# ======= Cres  = True; Decre = False
states = {'x': False, 'y': True}
statics = {'x': False, 'y': True}

# vel_y
#
# esq -
# dir +
#
# min -1.2
# max 1.2
#
# vel_x
#
# max 1.2
# min 0.0

# cres_decres = False # False True


vel = {}

y_factor = (vel_variation['y'][1] - vel_variation['y'][0])/((end - begin) + 2)

vel['y'] = np.arange(vel_variation['y'][0], vel_variation['y'][1], y_factor).tolist()

x_factor = (vel_variation['x'][1] - vel_variation['x'][0])/((end - begin) + 2)

vel['x'] = np.arange(vel_variation['x'][0], vel_variation['x'][1], x_factor).tolist()
# print(vel['x'], len(vel['x']))

for state, static in zip(states, statics):
    if states[state]:
        c = 0
    else:
        c = len(vel[state]) - 2
    for i in range(begin, end + 1):
        # print(df[state].at[i])
        if statics[static]:
            df[state].at[i] = float("%.4f" % random.uniform(vel_variation[state][0], vel_variation[state][1]))
        else:
            if states[state]:
                df[state].at[i] = float("%.4f" % random.uniform(vel[state][c], vel[state][c + 1]))
                c += 1
            else:
                df[state].at[i] = float("%.4f" % random.uniform(vel[state][c], vel[state][c + 1]))
                c -= 1

# c = 0
# for i in range(begin, end + 1):
#     if cres_decres:
#         df.at[i, 3] = float("%.4f" % random.uniform(vel_y[c], vel_y[c + 1]))
#         df.at[i, 4] = float("%.4f" % random.uniform(vel_x[c], vel_x[c + 1]))
#         c += 1
#     else:
#         df.at[i, 3] = float("%.4f" % random.uniform(vel_y_variation[0], vel_y_variation[1])) #desvio em yaw -1, 1
#         df.at[i, 4] = float("%.4f" % random.uniform(vel_x_variation[0], vel_x_variation[1])) #desvio em y -0.08, 0.08

print('Finalizou', begin, end)

df.to_csv(name, sep=" ", index=False, header=None)

