from dataset import Dataset
from file import File
import os
import pandas as pd

class Join(object):
    def __init__(self, dataset_dir, dataset_name, dataset_output_dir, output_dataset_name, datasets_list):
        self.datasets_list = datasets_list
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.dataset_output_dir = dataset_output_dir
        self.output_dataset_name = output_dataset_name
        self.dataset_all = None
        self.count = 0

    def appendSample(self, dataset_in, dataset):
        # print(self.dataset_all)
        for sample in range(len(dataset_in)):
            # print(self.count, dataset_in.iloc[sample])
            new_row = {'img_dataset': dataset_in['img_dataset'].iloc[sample],
                       'img_original': dataset_in['img_original'].iloc[sample],
                       'folder': dataset_in['folder'].iloc[sample],
                       'accx': dataset_in['accx'].iloc[sample], 'dataset': dataset, 'new_img_datset': self.convertIntToNameImg(self.count, ".jpg")}
            # print(new_row)
            self.dataset_all = self.dataset_all.append(new_row, ignore_index=True)
            self.count += 1

    def convertIntToNameImg(self, int, type):
        if int < 10:
            return "00000" + str(int) + type
        elif int < 100:
            return "0000" + str(int) + type
        elif int < 1000:
            return "000" + str(int) + type
        elif int < 10000:
            return "00" + str(int) + type
        elif int < 100000:
            return "0" + str(int) + type
        else:
            return str(int) + type

    def joinDatasets(self):
        self.dataset_all = pd.DataFrame(columns=["img_dataset", "img_original", "folder", "accx", 'dataset', 'new_img_datset'])
        for dataset in self.datasets_list:
            data = Dataset(self.dataset_dir, dataset, self.dataset_output_dir, self.output_dataset_name)
            data.loadDataset()
            self.appendSample(data.dataset, dataset)
            # print(data.dataset)
        f = File(self.dataset_dir)
        f.saveFileAllDataset(self.dataset_all, self.dataset_name)
        print(self.dataset_all)


dataset_directory = "../../datasets"
dataset_name = "market_accy_all_datasets_classes"
datasets_names_files = ['indoor_dataset_vely_all_out_classes_fliped', 'market_dataset_y_out_pc_classes_flipped']
output_dataset_name = "sidewalk_accy_158_pc_dataset_" # Not used

join = Join(dataset_directory, dataset_name, output_dataset_name, dataset_directory, datasets_names_files)
join.joinDatasets()