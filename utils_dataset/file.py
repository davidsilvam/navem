import os

class File(object):
    def __init__(self, path):
        self.path = path
        self.file = None

    def saveFile(self, dataset, folder):
        self.file = open(os.path.join(self.path, "gyro.txt"), "w")
        for sample in dataset:
            self.file.write(folder[0] + "/" + sample[0][:] + " " + str(sample[3]) + "\n")
            # print(folder[0] + "/" + sample[0][:] + " " + str(sample[3]) + "\n")
        self.file.close()
        print("File saved")

    def saveFile2(self, dataset, folder):
        self.file = open(os.path.join(self.path, "gyro.txt"), "w")
        for sample in range(len(dataset)):
            self.file.write(folder[0] + "/" + dataset['new_img_datset'].iloc[sample] + " " + str(dataset['accx'].iloc[sample]) + "\n")
            # print(folder[0] + "/" + sample[0][:] + " " + str(sample[3]) + "\n")
        self.file.close()
        print("File saved")

    def saveFileAllDataset(self, dataset, file_name):
        self.file = open(os.path.join(self.path, file_name + ".txt"), "w")
        for sample in range(len(dataset)):
            #print(dataset['img_dataset'].iloc[sample], dataset['img_original'].iloc[sample], dataset['folder'].iloc[sample])
            self.file.write(dataset['img_dataset'].iloc[sample] + " " +
                            dataset['img_original'].iloc[sample] + " " +
                            dataset['folder'].iloc[sample] + " " +
                            str(dataset['accx'].iloc[sample]) + " " +
                            dataset['dataset'].iloc[sample] + " " +
                            dataset['new_img_datset'].iloc[sample] + "\n")
        self.file.close()
