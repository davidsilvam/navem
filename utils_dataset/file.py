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