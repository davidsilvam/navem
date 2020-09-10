import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

from sklearn.metrics import r2_score

#predict_truth_train_model_weights_399.h5_0_
phase = "test"
set = "val"
exp = "exp_030"
name_weights = "model_weights_179.h5"

if(phase == "train"):
    name = os.path.join('./../experiments', exp, "predict_truth_" + set + "_model_" + name_weights + "_1_" + '.txt')
else:
    name = os.path.join('./../experiments', exp, "predict_truth_" + set + "_model_" + name_weights + "_0_" + '.txt')

df = pd.read_csv(name, sep=" ", engine="python", encoding="ISO-8859-1", names=['pred', 'real'])

print(r2_score(df['real'], df['pred']))

r2 = r2_score(df['real'], df['pred'])

plt.style.use('default')
plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(8, 4))

ax.scatter(df['real'], df['pred'], edgecolor='k', facecolor='grey', alpha=0.7, label='Sample data')

ax.set_ylabel('predict', fontsize=14)
ax.set_xlabel('real', fontsize=14)

ax.text(0.8, 0.1, 'aegis4048.github.io', fontsize=13, ha='center', va='center',
         transform=ax.transAxes, color='grey', alpha=0.5)
ax.legend(facecolor='white', fontsize=11)
ax.set_title('$%s -> R^2= %.2f$' % (set, r2), fontsize=18)

line = mlines.Line2D([0, 1], [0, 1], color='red')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.show()