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
        self.dataset_with_classes = None
        self.normalized_ = True
        self.classes = []
        self.last_file_name = None
        self.map_of_flipped_ = []
        self.debug = 0

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

    def AppendFlipSampleNewDataset(self, classe, sim_classe):
        max_samples = max(self.classes)
        mask_classe = self.dataset_with_classes['accx'] == classe
        mask_sim_classe = self.dataset_with_classes['accx'] == sim_classe
        df_classe = self.dataset_with_classes[mask_classe]
        df_sim_classe = self.dataset_with_classes[mask_sim_classe]
        # Insert samples of class sim_classe in classe
        for sample in range(df_sim_classe.shape[0]):
            if self.classes[classe] < max_samples:
                self.last_file_name += 1
                new_sample = pd.DataFrame(
                    [[str(self.FillZeros(str(self.last_file_name))) + ".jpg", df_sim_classe.iloc[sample]['img_original'], df_sim_classe.iloc[sample]['folder'],
                      classe, 1]], columns=['img_dataset', 'img_original', 'folder', 'accx', 'flag'])
                df_classe = df_classe.append(new_sample, ignore_index=True)
                self.classes[classe] += 1

                # self.map_of_flipped_.append(self.last_file_name)

                # print()
                # image = cv2.imread(os.path.join(self.dataset_dir, self.dataset_name, df_sim_classe.iloc[sample]['img_dataset']))
                # cv2.imwrite(
                #     os.path.join(self.dataset_dir, self.dataset_output_dir, str(self.FillZeros(str(self.last_file_name))) + ".jpg"),
                #     cv2.flip(image, 1))
                # self.debug += 1

        # Insert samples of class classe in sim_classe
        for sample in range(df_classe.shape[0]):
            if self.classes[sim_classe] < max_samples:
                self.last_file_name += 1
                new_sample = pd.DataFrame(
                    [[str(self.FillZeros(str(self.last_file_name))) + ".jpg", df_classe.iloc[sample]['img_original'], df_classe.iloc[sample]['folder'],
                      sim_classe, 1]], columns=['img_dataset', 'img_original', 'folder', 'accx', 'flag'])
                df_sim_classe = df_sim_classe.append(new_sample, ignore_index=True)
                self.classes[sim_classe] += 1

                # self.map_of_flipped_.append(self.last_file_name)

                # image = cv2.imread(os.path.join(self.dataset_dir, self.dataset_name, df_sim_classe.iloc[sample]['img_dataset']))
                # cv2.imwrite(
                #     os.path.join(self.dataset_dir, self.dataset_output_dir, str(self.FillZeros(str(self.last_file_name))) + ".jpg"),
                #     cv2.flip(image, 1))
                # self.debug += 1

        for sample in range(df_classe.shape[0]):
            new_sample = pd.DataFrame(
                [[df_classe.iloc[sample]['img_dataset'], df_classe.iloc[sample]['img_original'],
                  df_classe.iloc[sample]['folder'],
                  df_classe.iloc[sample]['accx'],
                  df_classe.iloc[sample]['flag']]], columns=['img_dataset', 'img_original', 'folder', 'accx', 'flag'])
            # print(new_sample)
            self.new_dataset = self.new_dataset.append(new_sample, ignore_index=True)
            # print(df_classe.iloc[sample]['img_dataset'], df_classe.iloc[sample]['img_original'], df_classe.iloc[sample]['folder'], df_classe.iloc[sample]['accx'])
            # if df_classe.iloc[sample]['flag'] == 1:
            #     image = cv2.imread(os.path.join(self.dataset_dir, self.dataset_name, df_classe.iloc[sample]['img_dataset']))
            #     cv2.imwrite(
            #         os.path.join(self.dataset_dir, self.dataset_output_dir, str(self.FillZeros(str(self.last_file_name))) + ".jpg"),
            #         cv2.flip(image, 1))
            # else:
            #     shutil.copyfile(
            #         os.path.join(self.dataset_dir, self.dataset_name, df_classe.iloc[sample]['img_dataset']),
            #         os.path.join(self.dataset_dir, self.dataset_output_dir, df_classe.iloc[sample]['img_dataset']))

            self.debug += 1

        if(classe != 2):
            for sample in range(df_sim_classe.shape[0]):
                new_sample = pd.DataFrame(
                    [[df_sim_classe.iloc[sample]['img_dataset'], df_sim_classe.iloc[sample]['img_original'],
                      df_sim_classe.iloc[sample]['folder'],
                      df_sim_classe.iloc[sample]['accx'],
                      df_sim_classe.iloc[sample]['flag']]], columns=['img_dataset', 'img_original', 'folder', 'accx', 'flag'])
                self.new_dataset = self.new_dataset.append(new_sample, ignore_index=True)
                # print(df_sim_classe.iloc[sample]['img_dataset'], df_sim_classe.iloc[sample]['img_original'],
                #       df_sim_classe.iloc[sample]['folder'], df_sim_classe.iloc[sample]['accx'])
                # if df_sim_classe.iloc[sample]['flag'] == 1:
                #     image = cv2.imread(
                #         os.path.join(self.dataset_dir, self.dataset_name, df_sim_classe.iloc[sample]['img_dataset']))
                #     cv2.imwrite(
                #         os.path.join(self.dataset_dir, self.dataset_output_dir,
                #                      str(self.FillZeros(str(self.last_file_name))) + ".jpg"),
                #         cv2.flip(image, 1))
                # else:
                #     shutil.copyfile(
                #         os.path.join(self.dataset_dir, self.dataset_name, df_sim_classe.iloc[sample]['img_dataset']),
                #         os.path.join(self.dataset_dir, self.dataset_output_dir,
                #                      df_sim_classe.iloc[sample]['img_dataset']))
                self.debug += 1

        # print(self.classes)
        print(df_classe.shape[0], df_sim_classe.shape[0])
        # os.system('pause')

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

    def createDatasetWithClasses(self):
        self.dataset_with_classes = pd.DataFrame()
        for sample in range(self.dataset.shape[0]):
            # print(self.dataset.iloc[sample]['img_dataset'], self.dataset.iloc[sample]['img_original'],
            #       self.dataset.iloc[sample]['folder'], self.dataset.iloc[sample]['accx'])
            new_sample = pd.DataFrame([[self.dataset.iloc[sample]['img_dataset'],
                                        self.dataset.iloc[sample]['img_original'], self.dataset.iloc[sample]['folder'],
                                        self.ConvertLabelsRegress2Classify(1, self.dataset.iloc[sample]['accx']), 0]], columns=['img_dataset',
                                                                                      'img_original', 'folder', 'accx', 'flag'])
            self.dataset_with_classes = self.dataset_with_classes.append(new_sample, ignore_index=True)
        self.classes = np.zeros(self.dataset_with_classes['accx'].nunique())
        for classe in range(len(self.classes)):
            mask = self.dataset_with_classes['accx'] == classe
            self.classes[classe] = int(self.dataset_with_classes[mask].shape[0])
        print(self.classes)

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
        self.last_file_name = int(self.raw_dataset.iloc[self.raw_dataset.shape[0] - 1]['img_dataset'].split('.')[0])

        # print(self.last_file_name)
        # os.system('pause')
        # for sample in range(self.dataset.shape[0]):
        if self.normalized_:
            self.createDatasetWithClasses()

        print('Started flipped images to classes 0 and 4')
        classe = {'0':4, '1': 3, '2': 2}
        # dictionary_items = classe.items()
        self.new_dataset = pd.DataFrame()
        for key, clas in classe.items():
            print(key, clas)
            Dataset.AppendFlipSampleNewDataset(self, int(key), int(clas))

        self.new_dataset.sort_values('img_dataset', inplace=True)
        self.new_dataset = self.new_dataset.reset_index(drop=True)

        print("Início")
        for sample in range(self.new_dataset.shape[0]):
            print(self.new_dataset.iloc[sample]['img_dataset'], self.new_dataset.iloc[sample]['img_original'],
                  self.new_dataset.iloc[sample]['folder'], self.new_dataset.iloc[sample]['accx'], self.new_dataset.iloc[sample]['flag'])
            if self.new_dataset.iloc[sample]['flag'] == 1:
                # print('asdf')
                mask_flip = (self.new_dataset['img_original'] == self.new_dataset.iloc[sample]['img_original']) & (self.new_dataset['folder'] == self.new_dataset.iloc[sample]['folder']) & (self.new_dataset['flag'] == 0)
                mask = (self.new_dataset['img_original'] == self.new_dataset.iloc[sample]['img_original']) & (
                            self.new_dataset['folder'] == self.new_dataset.iloc[sample]['folder']) & (
                                        self.new_dataset['flag'] == 1)
                data_flip = self.new_dataset[mask_flip]
                data = self.new_dataset[mask]
                # print(self.new_dataset[mask])
                # os.system('pause')
                # print(os.path.join(self.dataset_dir, self.dataset_name, str(data_flip['img_dataset'].item())))
                image = cv2.imread(
                    os.path.join(self.dataset_dir, self.dataset_name, str(data_flip['img_dataset'].item())))
                cv2.imwrite(
                    os.path.join(self.dataset_dir, self.dataset_output_dir,
                                 str(data['img_dataset'].item())),
                    cv2.flip(image, 1))
                # print(data_flip['img_dataset'].item(), data['img_dataset'].item())
                # os.system('pause')
            else:
                # print('copy only')
                shutil.copyfile(
                    os.path.join(self.dataset_dir, self.dataset_name, str(self.new_dataset.iloc[sample]['img_dataset'])),
                    os.path.join(self.dataset_dir, self.dataset_output_dir,
                                 str(self.new_dataset.iloc[sample]['img_dataset'])))


        print('Finish flipped images classes')
        print(self.new_dataset)
        # print(self.debug)
        os.system("pause")


    def Flip2(self):
        self.CreateDir()
        Dataset.LoadDataset(self)
        Dataset.LoadRawDataset(self)
        last_file_name = int(self.dataset.iloc[self.dataset.shape[0] - 1]['img_dataset'].split('.')[0])
        # for sample in range(self.dataset.shape[0]):
        if self.normalized_:
            self.createDatasetWithClasses()
        os.system("pause")
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
        # normalized_dataset = Dataset.NormalizeDataset(self, 'zero_one', self.new_dataset, 3)
        normalized_dataset = self.new_dataset.copy()
        normalized_dataset.sort_values('img_dataset', inplace=True)
        for sample in range(normalized_dataset.shape[0]):
            file_dataset.write(normalized_dataset.iloc[sample]['img_dataset'] + ' ' + normalized_dataset.iloc[sample]['img_original']
                               + ' ' + normalized_dataset.iloc[sample]['folder'] + ' ' +
                               str(normalized_dataset.iloc[sample]['accx']) + '\n')
        file_dataset.close()
        print('File saved')

    def SaveDatasetFliped2(self):
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
dataset_name = "indoor_dataset_vely_all_out_test"
raw_dataset_name = "indoor_dataset_vely"
output_dataset_name = "indoor_dataset_vely_all_out_classes_fliped"

fliped_dataset_name = "indoor_dataset_vely_all_out_classes_fliped"

flip = FlipImage(dataset_directory, dataset_name, raw_dataset_name, fliped_dataset_name, output_dataset_name)
flip.Flip()
flip.SaveDatasetFliped()
