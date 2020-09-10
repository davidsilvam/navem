import os
import pandas as pd

name = os.path.join('./../datasets', 'sidewalk_accx.txt')

df = pd.read_csv(name, sep=" ", engine="python", encoding="ISO-8859-1", header=None)

count_default = 0
count_deceleration = 0

for i in range(len(df)):
    if(df.at[i, 3] > 0.8):
        count_default+=1
    else:
        count_deceleration+=1

print("% walking - ", count_default / len(df), "%")
print("% deceleration - ", count_deceleration / len(df), "%")