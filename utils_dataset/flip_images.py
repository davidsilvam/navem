import os
import pandas as pd
import numpy as np
import cv2
import shutil


class Dataset(object):
    def __init__(self, dataset_dir, dataset_name, raw_dataset_name, dataset_output_dir, output_dataset_name):
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.dataset_output_dir = dataset_output_dir
        self.output_dataset_name = output_dataset_name
        self.raw_dataset_name = raw_dataset_name
        self.dataset = None
        self.raw_dataset = None
        self.new_dataset = None

    def LoadDataset(self):
        self.dataset = pd.read_csv(os.path.join(self.dataset_dir, self.dataset_name + ".txt"), sep=" ", engine="python",
                           encoding="ISO-8859-1", names=['img_dataset', 'img_original', 'folder', 'accx'])

    def LoadRawDataset(self):
        self.raw_dataset = pd.read_csv(os.path.join(self.dataset_dir, self.raw_dataset_name + ".txt"), sep=" ", engine="python",
                           encoding="ISO-8859-1", names=['img_dataset', 'img_original', 'folder', 'accx'])

    def AppendSampleNewDataset(self, sample, raw_sample):
        search = self.raw_dataset.loc[(self.raw_dataset['img_dataset'] == raw_sample)]
        if self.new_dataset is None:
            self.new_dataset = pd.DataFrame([[search['img_dataset'].item(), search['img_original'].item(), search['folder'].item(), search['accx'].item()]], columns=['img_dataset', 'img_original', 'folder', 'accx'])
        else:
            if(sample == raw_sample):
                new_sample = pd.DataFrame([[search['img_dataset'].item(), search['img_original'].item(), search['folder'].item(),
                                   search['accx'].item()]], columns=['img_dataset', 'img_original', 'folder', 'accx'])
            else:
                new_sample = pd.DataFrame([[search['img_dataset'].item(), search['img_original'].item(), search['folder'].item(),
                                   search['accx'].item()]], columns=['img_dataset', 'img_original', 'folder', 'accx'])
                self.new_dataset = self.new_dataset.append(new_sample, ignore_index=True)
                new_sample = pd.DataFrame([[self.FillZeros(str(sample)) + ".jpg", search['img_original'].item(), search['folder'].item(),
                                   search['accx'].item()*-1]], columns=['img_dataset', 'img_original', 'folder', 'accx'])
            self.new_dataset = self.new_dataset.append(new_sample, ignore_index=True)

    def FillZeros(self, value):
        v = 6 - self.GetSizeValue(value)

        if v == 0:
            return value
        elif v == 1:
            return "0" + value
        elif v == 2:
            return "00" + value
        elif v == 3:
            return "000" + value
        elif v == 4:
            return "0000" + value
        else:
            return "00000" + value

    def ConvertLabelsRegress2Classify(self, maximun, value):
        if (value > 0) and (value <= maximun / 5):
            return 0
        elif (value > maximun / 5) and (value <= (maximun / 5) * 2):
            return 1
        elif (value > (maximun / 5) * 2) and (value <= (maximun / 5) * 3):
            return 2
        elif (value > (maximun / 5) * 3) and (value <= (maximun / 5) * 4):
            return 3
        else:
            return 4

    def NormalizeDataset(self, method, data, position):
        dataset_temp = self.new_dataset.copy()
        if method == 'zero_one':
            dataset_temp['accx'] = (data['accx'] - data['accx'].min()) / (data['accx'].max() - data['accx'].min())
            return dataset_temp
        elif method == 'std_mean':
            dataset_temp['accx'] = (data['accx'] - np.mean(data['accx']))/np.std(data['accx'])
            return dataset_temp

class File(object):
    def __init__(self):
        pass


class FlipImage(Dataset):
    def __init__(self, dataset_dir, dataset_name, raw_dataset_name, dataset_output_dir, output_dataset_name):
        Dataset.__init__(self, dataset_dir, dataset_name, raw_dataset_name, dataset_output_dir, output_dataset_name)

    def CreateDir(self):
        if not os.path.exists(os.path.join(self.dataset_dir, self.dataset_output_dir)):
            os.makedirs(os.path.join(self.dataset_dir, self.dataset_output_dir))
            print("Path", os.path.join(self.dataset_dir, self.dataset_output_dir), "created.")
        else:
            print("Directory '{dir}' already exist.".format(dir = os.path.join(self.dataset_dir,
                                                                               self.dataset_output_dir)))

    def GetSizeValue(self, value):
        v = int(value)
        c = 0
        for i in range(len(value)):
            v /= 10
            c += 1
        return c

    def Flip(self):
        self.CreateDir()
        Dataset.LoadDataset(self)
        Dataset.LoadRawDataset(self)
        last_file_name = int(self.dataset.iloc[self.dataset.shape[0] - 1]['img_dataset'].split('.')[0])
        # for sample in range(self.dataset.shape[0]):
        print('Started flipped images to classes 0 and 4')
        for sample in range(self.dataset.shape[0]):
            # print(self.dataset['accx'][sample], Dataset.ConvertLabelsRegress2Classify(self, 1, self.dataset['accx'][sample]))
            label = Dataset.ConvertLabelsRegress2Classify(self, 1, self.dataset['accx'][sample])
            if(label == 0 or label == 4):
                last_file_name += 1
                # print(str(self.FillZeros(str(last_file_name))) + ".jpg")
                image = cv2.imread(os.path.join(self.dataset_dir, self.dataset_name, self.dataset.iloc[sample]['img_dataset']))
                cv2.imwrite(os.path.join(self.dataset_dir, self.dataset_output_dir, str(self.FillZeros(str(last_file_name))) + ".jpg"), cv2.flip(image, 1))
                shutil.copyfile(os.path.join(self.dataset_dir, self.dataset_name, self.dataset.iloc[sample]['img_dataset']),
                                os.path.join(self.dataset_dir, self.dataset_output_dir, self.dataset.iloc[sample]['img_dataset']))
                Dataset.AppendSampleNewDataset(self, last_file_name, self.dataset.iloc[sample]['img_dataset'])
            else:
                Dataset.AppendSampleNewDataset(self, self.dataset.iloc[sample]['img_dataset'],
                                               self.dataset.iloc[sample]['img_dataset'])
                shutil.copyfile(os.path.join(self.dataset_dir, self.dataset_name, self.dataset.iloc[sample]['img_dataset']),
                                os.path.join(self.dataset_dir, self.dataset_output_dir, self.dataset.iloc[sample]['img_dataset']))
        # print(self.new_dataset)
        print('Finish flipped images classes')

    def SaveDatasetFliped(self):
        print('Started create file and write with new dataset')
        file_dataset = open(os.path.join(self.dataset_dir, self.output_dataset_name + ".txt"), "w")
        normalized_dataset = Dataset.NormalizeDataset(self, 'zero_one', self.new_dataset, 3)
        normalized_dataset.sort_values('img_dataset', inplace=True)
        for sample in range(normalized_dataset.shape[0] - 1):
            file_dataset.write(normalized_dataset.iloc[sample]['img_dataset'] + ' ' + normalized_dataset.iloc[sample]['img_original']
                               + ' ' + normalized_dataset.iloc[sample]['folder'] + ' ' +
                               str(normalized_dataset.iloc[sample]['accx']) + '\n')
        file_dataset.close()
        print('File saved')


dataset_directory = "../../datasets"
dataset_name = "sidewalk_accy_proportion_classes"
raw_dataset_name = "sidewalk_accy"
output_dataset_name = "sidewalk_accy_proportion_classes_fliped"

fliped_dataset_name = "sidewalk_accy_fliped"

flip = FlipImage(dataset_directory, dataset_name, raw_dataset_name, fliped_dataset_name, output_dataset_name)
flip.Flip()
flip.SaveDatasetFliped()