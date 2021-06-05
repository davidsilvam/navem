import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
##import scipy.stats as stats
##import pylab as pl

dataset_directory = "./../../datasets"
o_dataset_name = "indoor_dataset_velx"

data = pd.read_csv(os.path.join(dataset_directory, o_dataset_name + ".txt"), sep=" ", engine="python", encoding="ISO-8859-1", header=None).values
    

##norm_m = (data[:,2] - np.mean(data[:,2]))/np.std(data[:,2])#normalize (x - media) / desvio padrao
##norm = (data[:,2]-data[:,2].min())/(data[:,2].max()-data[:,2].min())
col = 3
d = data[:,col]
d1 = (data[:,col]-data[:,col].min())/(data[:,col].max()-data[:,col].min())
d2 = (data[:,col] - np.mean(data[:,col]))/np.std(data[:,col])#normalize (x - media) / desvio padrao

#########
c = [0, 0, 0, 0, 0]
for i in d1:    
    if i > 0 and i <= 0.2:
        c[0]+=1
    if i > 0.2 and i <= 0.4:
        c[1]+=1
    if i > 0.4 and i <= 0.6:
        c[2]+=1
    if i > 0.6 and i <= 0.8:
        c[3]+=1
    if i > 0.8 and i <= 1:
        c[4]+=1
##    print(i)

print(max(d1))
print(np.where(d1 == 1))
print(c)

#########

##plt.hist(d, 30, alpha=0.5, edgecolor='black', color='red', linewidth=1.2, label='Raw data')
##plt.plot()
####plt.hist(d1, 30, alpha=0.5, edgecolor='black', color='yellow', linewidth=1.2, label='Norm 0 to 1')
####plt.hist(d2, 30, alpha=0.5, edgecolor='black', color=(0, 0, 0.5), linewidth=1.2, label='Norm (x - mean)/std')
####pyplot.hist(y, 10, alpha=0.5, label='y')
##plt.legend(loc='upper right')
##plt.show()

fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharey=True, dpi=100)
##plt.figure(figsize=(10,7), dpi= 100)
sns.distplot(d , ax=axes[0], bins = 30 , kde_kws={"color": "g", "lw": 1, "label": "Raw data"} ,hist_kws={"linewidth": 1.3, "alpha": 0.5, "color": "g", "edgecolor":"black"})
sns.distplot(d1 , ax=axes[1], bins = 30 , kde_kws={"color": "b", "lw": 1, "label": "Norm 0-1"} ,hist_kws={"linewidth": 1.3, "alpha": 0.5, "color": "b", "edgecolor":"black"})
sns.distplot(d2 , ax=axes[2], bins = 30 , kde_kws={"color": "r", "lw": 1, "label": "Norm (x - mean)/std"} ,hist_kws={"linewidth": 1.3, "alpha": 0.5, "color": "r", "edgecolor":"black"})
##plt.plot(alpha=0.5, edgecolor='black', linewidth=1.2)
##sns.distplot(x2 , color="deeppink", ax=axes[1], axlabel='Fair')
##sns.distplot(x3 , color="gold", ax=axes[2], axlabel='Good')
##plt.xlim(min(d),max(d))
plt.legend();
plt.show()
