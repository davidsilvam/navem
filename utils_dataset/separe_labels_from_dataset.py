import os
import pandas as pd

name = os.path.join('./../../datasets', 'market_dataset_2_xy.txt')

df = pd.read_csv(name, sep=" ", engine="python", encoding="ISO-8859-1", header=None)

f = open(os.path.join('./../../datasets', 'market_dataset_2_y.txt'), "w")

for i in range(len(df)):
    # f.write(str(df.at[i, 0]) + " " + str(df.at[i, 1]) + " " + str(df.at[i, 2]) + " " + str(df.at[i, 3]) + "\n") # for only x
    f.write(str(df.at[i, 0]) + " " + str(df.at[i, 1]) + " " + str(df.at[i, 2]) + " " + str(df.at[i, 4]) + "\n")  # for only y
    # f.write(str(df.at[i, 0]) + " " + str(df.at[i, 1]) + " " + str(df.at[i, 2]) + " " + str(df.at[i, 3]) + " " + str(df.at[i, 4]) + "\n")
    # print(str(df.at[i, 0]) + " " + str(df.at[i, 1]) + " " + str(df.at[i, 2]) + " " + str(df.at[i, 3]) + " " + str(df.at[i, 3]) + "\n")
f.close()