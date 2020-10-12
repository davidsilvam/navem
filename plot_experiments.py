import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

#dronet = ["exp_001", "exp_002"]
#"exp_020", "exp_021"
#"exp_022", "exp_023", "exp_024"
#"exp_025"
#"exp_026"
#
vgg16 = ["exp_040"]#side, inner, external
names = ["vgg16"]
exp_name = vgg16
#data = pd.read_csv("/home/david/Área de Trabalho/navem_keras/experiments/" + exp_name + "/log.txt", sep="\t", engine="python", encoding="ISO-8859-1", header=None)

#plt.ylim(-2, 2)

for exp, name in zip([vgg16], names):
    plt.figure("vgg16")

    for i in enumerate(exp):
        dir = "./../experiments/" + i[1]
        #dir = "/home/david/Área de Trabalho/navem_keras/arquivos/AWS/experimets/" + i[1]
        log_file = os.path.join(dir, "log.txt")
        log = np.genfromtxt(log_file, delimiter='\t',dtype=None, names=True)

        train_loss = log['train_loss']
        val_loss = log['val_loss']
        timesteps = list(range(train_loss.shape[0]))
        plt.title(name)
        plt.subplot(len(exp_name), 1, i[0] + 1)
        #plt.ylim(0, 0.05)
        # Plot losses
        plt.plot(timesteps, train_loss, 'r-', timesteps, val_loss, 'b-')
        plt.legend(["Training loss", "Validation loss"])
        plt.ylabel('Loss')
        plt.xlabel('Epochs')

plt.show()
#print(len(data))
#print(data[0])
#epochs = np.arange(len(data))
#print(epochs, len(data[0]))
#print(data["train_loss"])

#plt.plot(epochs, data[0], 'b', label='Training loss')
#plt.plot(epochs, data[1], 'r', label='Validation loss')
#plt.title('Training and validation loss')
#plt.legend()
#plt.show()
