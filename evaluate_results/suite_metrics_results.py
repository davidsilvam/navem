import os
import pandas as pd
import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2
import pickle
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from scipy import stats

class ResultsExperiments():
    def __init__(self, exps_path, exps, phase, name_weigths, set, num_classes, load_bests=False, percent=False):
        self._exps_path = exps_path
        self._exps = exps
        self._array_raw_exps = []
        self._phase = phase
        self._num_classes = num_classes
        self._set = set
        self._name_weights = name_weigths
        self._all_accuracy = []
        self._all_precision = []
        self._all_recall = []
        self._all_F1_score = []
        self._all_confusion_matrix = []
        self._best_exps = {}
        self._best_results = {}
        self._load_best_models = load_bests # Means that load weights are the best fo all models
        self.percent = percent

    def load_results_file(self):
        """
        Load file from results of experiments
        :return:
        """
        for exp in self._exps:
            name = ""
            if self._load_best_models:
                name_weights = "model_weights_" + str(self._best_exps[str(exp)]) + ".h5"
            else:
                name_weights = self._name_weights
            if self._phase == "train":
                name = os.path.join(self._exps_path, "exp_" + str(exp),
                                      "predict_truth_" + set + "_" + name_weights + "_1_" + ".txt")
            else:
                name = os.path.join(self._exps_path, "exp_" + str(exp),
                                      "predict_truth_" + set + "_" + name_weights + "_0_" + ".txt")
            df = pd.read_csv(name, sep=" ", engine="python", encoding="ISO-8859-1", names=['pred', 'real'])
            self._array_raw_exps.append(df)

    def print_best_result(self):
        """
        Print best results from each experiments
        :return:
        """
        for exp in self._best_results:
            # print(exp, self._best_results[exp])
            print("============== Exp {0} ==================".format(exp))
            for metric in self._best_results[exp]:
                print(metric, self._best_results[exp][metric])

    def create_dict_best_model(self, dict, metric):
        """
        Create dictionary from metric use to logs
        :param dict: Dictionary initialized
        :param metric: Metric name
        :return: Dictionary filled with metrics
        """
        if metric == "train_loss" or metric == "val_loss":
            res = {"best_all": 999, "index_best_all": -1, "best_10_by_10": 999, "index_best_10by_10": -1}
        else:
            res = {"best_all": -1, "index_best_all": -1, "best_10_by_10": -1, "index_best_10by_10": -1}
        dict[metric] = res
        return dict

    def add_best_exp(self, exp, array_best, metric="train_loss"):
        """
        Add best result of a metric in dictionary _best_exps
        :param exp: Experiment number
        :param array_best: Dictionary with best results
        :param metric: Metric name
        :return:
        """
        self._best_exps[exp] = array_best[metric]["index_best_10by_10"]

    def print_best_exps(self, metric="acc_loss"):
        """
        Print best experiments
        :param metric: Metric name
        :return:
        """
        for exp in self._best_results:
            self.add_best_exp(exp, self._best_results[exp], metric)
        print("============ Best exps ============")
        print(self._best_exps)

    def best_result(self, exp, array_log):
        """
        Get best results from each experiment
        :param exp:
        :param array_log:
        :return:
        """
        max = {}
        metrics = ["train_loss", "acc", "val_loss", "acc_loss"]
        for metric in metrics:
            max = self.create_dict_best_model(max, metric)
            for value, res in enumerate(array_log[metric]):
                if metric == "train_loss" or metric == "val_loss":
                    if max[metric]["best_all"] > res:
                        max[metric]["best_all"] = res
                    # print(value % 10, max[metric]["best_10_by_10"] > res, value )
                    if value % 10 == 9 and max[metric]["best_10_by_10"] > res and value != 0:
                        max[metric]["best_10_by_10"] = res
                        max[metric]["index_best_10by_10"] = value
                else:
                    if max[metric]["best_all"] < res:
                        max[metric]["best_all"] = res
                    if value % 10 == 9 and max[metric]["best_10_by_10"] < res and value != 0:
                        max[metric]["best_10_by_10"] = res
                        max[metric]["index_best_10by_10"] = value
            # print(max[metric])
            max[metric]["index_best_all"] = np.where(array_log[metric] == max[metric]["best_all"])[0][0]
            # max[metric]["index_best_10by_10"] = np.where(array_log[metric] == max[metric]["best_10_by_10"])[0][0]
        self._best_results[str(exp)] = max
        # self.print_best_result(max)

    def get_results_log(self, exp, file="log.txt"):
        """
        Load and get run best results
        :param exp:
        :param file:
        :return:
        """
        for e in exp:
            log = np.genfromtxt(os.path.join(self._exps_path, "exp_" + str(e), file), delimiter='\t', dtype=None, names=True)
            # train_loss = log['train_loss']
            # val_loss = log['val_loss']
            # acc = log['acc']
            # val_acc = log['acc_loss']
            self.best_result(e, log)
            # print("================= Exp {0} =================".format(e))

    def get_confusion_matrix(self, num_classes, df_exp):
        """
        Fill an array with confusion matrix of an experiment
        :param num_classes: Quantity of classes have in experiment
        :param df_exp: Dataframe loaded from experiment
        :return: Confusion matrix filled
        """
        matrix = np.zeros((num_classes, num_classes))
        for i in range(len(df_exp)):
            matrix[int(df_exp['real'][i])][int(df_exp['pred'][i])] += 1
        return matrix

    def confusion_matrix(self):
        """
        Create confusion matrix for whole experiments passed on this suite
        :return:
        """
        for exp in self._array_raw_exps:
            self._all_confusion_matrix.append(self.get_confusion_matrix(self._num_classes, exp))
            # print(self._all_confusion_matrix[0])
            # os.system("pause")

    def accuracy(self):
        """
        Fill array with all accuracy of experiments
        :return:
        """
        for matrix in self._all_confusion_matrix:
            self._all_accuracy.append(sum(np.diag(matrix)) / np.sum(matrix))

    def get_precision(self, matrix):
        """
        Obtain precision from whole classes of an experiments and append in array with all experiments
        :param matrix: Confusion matrix of an experiment
        :return: Array filled with precisions for each class
        """
        precision_classes = []
        for clas in range(len(matrix)):
            precision_classes.append(self.precision_class(clas, matrix))
        return precision_classes

    def precision(self):
        """
        Fill array with all precisions of experiments
        :return:
        """
        for matrix in self._all_confusion_matrix:
            self._all_precision.append(self.get_precision(matrix))

    def get_recall(self, matrix):
        """
        Obtain recall from whole classes of an experiments and append in array with all experiments
        :param matrix: Confusion matrix of an experiment
        :return: Array filled with precisions for each class
        """
        recall_classes = []
        for clas in range(len(matrix)):
            recall_classes.append(self.recall_class(clas, matrix))
        return recall_classes

    def recall(self):
        """
        Fill array with all recall of experiments
        :return:
        """
        for matrix in self._all_confusion_matrix:
            self._all_recall.append(self.get_recall(matrix))

    def precision_class(self, clas, matrix):
        """
        Calculate the precision from a class
        :param clas: Class number wich will be calculated
        :param matrix: Matrix confusion from specific experiments
        :return: Result of the precision
        """
        return matrix[clas][clas] / np.sum(matrix[clas, :])

    def recall_class(self, clas, matrix):
        """
        Calculate the recall from a class
        :param clas: Class number wich will be calculated
        :param matrix: Matrix confusion from specific experiments
        :return: Result of the recall
        """
        return matrix[clas][clas] / np.sum(matrix[:, clas])

    def get_f1_score(self, matrix):
        """
        Calculate F1 score from specific experiment
        :param matrix: Confusion matrix of the experiment
        :return: Array filled with F1 score of each class
        """
        f1_score_classes = []
        # print(matrix)
        for clas in range(len(matrix)):
            # print((2 * self.precision_class(clas, matrix) * self.recall_class(clas, matrix)))
            # print((self.precision_class(clas, matrix)))
            # print((self.recall_class(clas, matrix)), clas)
            # os.system("pause")
            f1_score_classes.append((2 * self.precision_class(clas, matrix) * self.recall_class(clas, matrix)) / (self.precision_class(clas, matrix) + self.recall_class(clas, matrix)))
        # np.seterr('raise')
        return f1_score_classes

    def f1_score(self):
        """
        Obtain F1 score from whole experiments
        :return:
        """
        c = 0
        for matrix in self._all_confusion_matrix:
            # print("========== {0} ========".format(c))
            self._all_F1_score.append(self.get_f1_score(matrix))
            c = c + 1

    def get_metrics(self):
        """
        Obtain all metrics of a determinate experiment
        :return:
        """
        self.load_results_file()
        self.confusion_matrix()
        self.accuracy()
        self.precision()
        self.recall()
        self.f1_score()

    def confusion_matrix_as_percentage(self, array):
        percent_matrix = []
        # print(array, len(array))

        for i in range(len(array)):
            a = sum(array[i])
            per = []
            for j in range(len(array[0])):
                per.append(array[i][j]/a)
            percent_matrix.append(per)
        return np.array(percent_matrix)

    def format_print_percent(self, matrix):
        # print(matrix)
        a = matrix
        for elem in a:
            # print(elem)
            for acc in elem:
                print('{acc:0>5.2f}'.format(acc=(acc*100)), end=' ')
            print()

    def print_mean_std_precision_all_classes(self):
        all_exps_precisions = pd.DataFrame(columns=["class_0", "class_1", "class_2", "class_3", 'class_4'])
        for exp in range(len(self._exps)):
            exp_values = self._all_precision[exp]
            exp = {"class_0": exp_values[0], "class_1": exp_values[1], "class_2": exp_values[2],
                   "class_3": exp_values[3], "class_4": exp_values[4]}
            all_exps_precisions = all_exps_precisions.append(exp, ignore_index=True)
            # print(self._all_precision[exp])
        for value, clas in enumerate(all_exps_precisions.keys()):
            print("{clas:<6d} {mean:.5f}\xB1{std:.5f}".format(clas=value, mean=np.mean(all_exps_precisions[clas]), std=np.std(all_exps_precisions[clas])))

    def std_matrix(self, mean):
        result_matrix = (self._all_confusion_matrix[0] - mean)**2
        for exp in range(1,len(self._exps)):
            result_matrix = result_matrix + (self._all_confusion_matrix[exp] - mean)**2
        return np.sqrt(result_matrix / len(self._exps))

    def print_format_all_matrix(self, mean_marix, std_matrix):
        for i in range(self._num_classes):
            for j in range(self._num_classes):
                print('{mean:0>7.2f}\xB1{std:0>8.5f}'.format(mean=mean_marix[i][j], std=std_matrix[i][j]), end=' ')
            print()

    def print_format_all_matrix_as_percent(self, mean_marix, std_matrix):
        for i in range(self._num_classes):
            for j in range(self._num_classes):
                print('{mean:0>5.5f}\xB1{std:.5f}'.format(mean=mean_marix[i][j], std=std_matrix[i][j]), end=' ')
            print()

    def confusion_all_matrix_as_percentage(self, array, as_std=False, mean_array=None):
        percent_matrix = []
        # print(array, len(array))

        for i in range(len(array)):
            if as_std:
                a = sum(mean_array[i])
            else:
                a = sum(array[i])
            per = []
            for j in range(len(array[0])):
                per.append(array[i][j]/a)
            percent_matrix.append(per)
        return np.array(percent_matrix)

    def print_mean_matrix_confusion(self, as_percent=False):
        np.set_printoptions(suppress=True)

        result_matrix = self._all_confusion_matrix[0]
        for exp in range(1,len(self._exps)):
            result_matrix = result_matrix + self._all_confusion_matrix[exp]
        mean_matrix = result_matrix / len(self._exps)
        std_matrix = self.std_matrix(mean_matrix)

        # for i in range(self._num_classes):
        #     for j in range(self._num_classes):
        #         print('{mean:0>7.2f}\xB1{std:.5f}'.format(mean=mean_matrix[i][j], std=std_matrix[i][j]), end=' ')
        #     print()

        if not as_percent:
            self.print_format_all_matrix(mean_matrix, std_matrix)
        else:
            percent_mean_all_matrix = self.confusion_matrix_as_percentage(mean_matrix)
            percent_sdt_all_matrix = self.confusion_all_matrix_as_percentage(std_matrix, as_std=True, mean_array=mean_matrix)

            self.print_format_all_matrix_as_percent(percent_mean_all_matrix, percent_sdt_all_matrix)

    def generate_statistics(self):
        """
        Generate all results of statistics and get best model
        :return:
        """
        self.get_metrics()
        mean_accuracy = self.mean_accuracy(self._all_accuracy)
        mean_precision = self.mean_precision(self._all_precision)
        print("======== Metrics ===========")
        print("{:10} {:6} {:5}".format('metric', 'mean', 'std'))
        print("{metric:10} {mean:.5f}\xB1{std:.5f}".format(metric='accuracy',mean=mean_accuracy[0], std=mean_accuracy[1]))
        print("{metric:10} {mean:.5f}\xB1{std:.5f}".format(metric='precision',mean=mean_precision[0], std=mean_precision[1]))
        print("{metric:10} {mean:.5f}\xB1{std:.5f}".format(metric='recall', mean=self.mean_recall(self._all_recall)[0], std=self.mean_recall(self._all_recall)[1]))
        print("{metric:10} {mean:.5f}\xB1{std:.5f}".format(metric='f1_score', mean=self.mean_f1_score(self._all_F1_score)[0], std=self.mean_f1_score(self._all_F1_score)[1]))
        print("-------- precision ---------")
        print("{:6} {:7} {:5}".format('class', 'mean', 'std'))
        self.print_mean_std_precision_all_classes()
        print("------------ mean all confusion matrix ------------")
        self.print_mean_matrix_confusion()
        print("--------- mean all confusion matrix as percent -----")
        self.print_mean_matrix_confusion(as_percent=True)

        best_model = self.get_best_model("accuracy")
        print("======= Best model =========")
        print("Exp {0}".format(self._exps[best_model["index"]]))
        print("------ Accuracy -------")
        print("Accuracy -> {mean:.5f}".format(mean=best_model["max_value"]))
        if self.percent:
            # print(self.confusion_matrix_as_percentage(self._all_confusion_matrix[best_model["index"]]))
            self.format_print_percent(self.confusion_matrix_as_percentage(self._all_confusion_matrix[best_model["index"]]))
        #else:
        print(self._all_confusion_matrix[best_model["index"]])
        print("------ Precision -------")
        self.print_metric_by_class(self._all_precision[best_model["index"]])
        print("Mean precision -> {mean:.5f}".format(mean=np.mean(self._all_precision[best_model["index"]])))
        print("------ Recall -------")
        self.print_metric_by_class(self._all_recall[best_model["index"]])
        print("Mean recall -> {mean:.5f}".format(mean=np.mean(self._all_recall[best_model["index"]])))
        print("------ F1 score ------")
        self.print_metric_by_class(self._all_F1_score[best_model["index"]])
        print("Mean F1 score -> {mean:.5f}".format(mean=np.mean(self._all_F1_score[best_model["index"]])))

    def print_metric_by_class(self, matrix):
        """
        Print results per class
        :param matrix: Confusion matrix
        :return:
        """
        for value, clas in enumerate(matrix):
            print("Class {0} -> {acc: .5f}".format(value, acc=clas))

    def mean_accuracy(self, array_accuracy):
        """
        Calculate mean and standard deviation from accuracy array
        :param array_accuracy: Arrau content results accuracy of an experiments
        :return: Tuple content result of the mean and standard deviation
        """
        return np.mean(array_accuracy), np.std(array_accuracy)

    def mean_precision(self, array_precision):
        """
        Calculate mean precision and standard deviation of classes of experiments
        :param array_precision: Array content one or more experiments results of precision
        :return: Tuple content result of the mean and standard deviation
        """
        mean_classes_precision_exp = []
        for exp in array_precision:
            mean_classes_precision_exp.append(np.mean(exp))
        return np.mean(mean_classes_precision_exp), np.std(mean_classes_precision_exp)

    def mean_recall(self, array_recall):
        """
        Calculate mean recall and standard deviation of classes of experiments
        :param array_recall: Array content one or more experiments results of recall
        :return: Tuple content result of the mean and standard deviation
        """
        mean_classes_recall_exp = []
        for exp in array_recall:
            mean_classes_recall_exp.append(np.mean(exp))
        return np.mean(mean_classes_recall_exp), np.std(mean_classes_recall_exp)

    def mean_f1_score(self, array_f1_score):
        """
        Calculate mean f1 score and standard deviation of classes of experiments
        :param array_recall: Array content one or more experiments results of f1 score
        :return: Tuple content result of the mean and standard deviation
        """
        mean_classes_f1_score_exp = []
        for exp in array_f1_score:
            mean_classes_f1_score_exp.append(np.mean(exp))
        return np.mean(mean_classes_f1_score_exp), np.std(mean_classes_f1_score_exp)

    def get_best_model(self, metric):
        """
        Get best model of according to metric
        :param metric: string metric
        :return: Confution matrix of best model
        """
        self.get_metrics()
        max = -1
        if(metric == "accuracy"):
            for acc in self._all_accuracy:
                if max < acc:
                  max = acc
            return {"max_value": max, "index": self._all_accuracy.index(max)}

    def mse(self, imageA, imageB):
        # the 'Mean Squared Error' between the two images is the
        # sum of the squared difference between the two images;
        # NOTE: the two images must have the same dimension
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])

        # return the MSE, the lower the error, the more "similar"
        # the two images are
        return err

    def get_ssmi_by_classes(self, path):
        a = self.get_best_model('accuracy')
        df_images = pd.read_csv(os.path.join(path, 'gyro.txt'), sep=" ", engine="python", encoding="ISO-8859-1", names=['path_img', 'real'])
        # print(a)
        # print(self._exps[a['index']])
        # print(self._array_raw_exps[a['index']])# Criar estrutura de dados com os erros entre classes
        row_class_matrix = {'0': {'0': [], '1': [], '2': [], '3': [], '4': []},
                            '1': {'0': [], '1': [], '2': [], '3': [], '4': []},
                            '2': {'0': [], '1': [], '2': [], '3': [], '4': []},
                            '3': {'0': [], '1': [], '2': [], '3': [], '4': []},
                            '4': {'0': [], '1': [], '2': [], '3': [], '4': []}
                            }
        num_classes = 5
        df_exp = self._array_raw_exps[a['index']]
        # print(df_exp)
        # print(df_exp.shape[0])
        # row_class_matrix['0']['0'].append(1)
        # print(row_class_matrix)
        # print(df_exp.shape[0])
        # os.system('pause')
        for i in range(df_exp.shape[0]):
            # matrix[int(df_exp['real'][i])][int(df_exp['pred'][i])] += 1
            # print(i)
            row_class_matrix[str(int(df_exp['real'][i]))][str(int(df_exp['pred'][i]))].append(i)
        # print(len(row_class_matrix['0']['1']))
        # print(df_images)

        c_inv = 1
        # print(len(row_class_matrix[str(c_inv - 1)][str(c_inv - 1)]))
        # print(len(row_class_matrix[str(c_inv)][str(c_inv - 1)]))

        c1 = []
        c1_df = pd.DataFrame(columns=["class_middle", "class_neibor", "img_midle", "img_neibor", 'ssim', 'mse'])
        c2 = []
        c2_df = pd.DataFrame(columns=["class_middle", "class_neibor", "img_midle", "img_neibor", 'ssim', 'mse'])

        for row in row_class_matrix[str(c_inv - 1)][str(c_inv - 1)]:
            # print(row)
            # r = 1
            for row_inv in row_class_matrix[str(c_inv)][str(c_inv - 1)]:
                # print(row_inv)
                # r = 3
                # print(row_inv, row)
                path_img_1 = os.path.join(path, 'images', df_images['path_img'].iloc[row_inv].split('/')[1])
                path_img_2 = os.path.join(path, 'images', df_images['path_img'].iloc[row].split('/')[1])
                img1 = cv2.cvtColor(cv2.imread(path_img_1), cv2.COLOR_BGR2GRAY)
                img2 = cv2.cvtColor(cv2.imread(path_img_2), cv2.COLOR_BGR2GRAY)
                ssim_imgs = ssim(img1, img2)
                mse_imgs = self.mse(img1, img2)
                c1.append(ssim)
                new = {"class_middle": c_inv, "class_neibor": c_inv - 1, "img_midle": path_img_1,"img_neibor": path_img_2, "ssim": ssim_imgs, "mse": mse_imgs}
                c1_df = c1_df.append(new, ignore_index=True)
                # print('p1 pass', row_inv, row)
                # print(c1)
        print('==================')
        for row in row_class_matrix[str(c_inv + 1)][str(c_inv + 1)]:
            # print(row)
            # r = 1
            for row_inv in row_class_matrix[str(c_inv)][str(c_inv + 1)]:
                # print(row_inv)
                # r = 3
                # print(row_inv, row)
                path_img_1 = os.path.join(path, 'images', df_images['path_img'].iloc[row_inv].split('/')[1])
                path_img_2 = os.path.join(path, 'images', df_images['path_img'].iloc[row].split('/')[1])
                # print('pass')
                img1 = cv2.cvtColor(cv2.imread(path_img_1), cv2.COLOR_BGR2GRAY)
                img2 = cv2.cvtColor(cv2.imread(path_img_2), cv2.COLOR_BGR2GRAY)
                ssim_imgs = ssim(img1, img2)
                mse_imgs = self.mse(img1, img2)
                c2.append(ssim_imgs)
                new = {"class_middle": c_inv, "class_neibor": c_inv + 1, "img_midle": path_img_1,
                       "img_neibor": path_img_2, "ssim": ssim_imgs, "mse": mse_imgs}
                c2_df = c2_df.append(new, ignore_index=True)
                # print('p2 pass', row_inv, row)
                # print(c1)
        # teoricamente afzer o mean
        print('aqui')
        print(len(c1), len(c2))

        print("ssim")
        print('mean {0} vs {1} -> ssim {2} std -> {3} max -> {4}'.format(c_inv, c_inv - 1, np.mean(c1_df['ssim']),
                                                                         np.std(np.array(c1_df['ssim']), axis=0),
                                                                         c1_df['ssim'].max()))
        print('mean {0} vs {1} -> ssim {2} std -> {3} max -> {4}'.format(c_inv, c_inv + 1, np.mean(c2_df['ssim']),
                                                                         np.std(np.array(c2_df['ssim']), axis=0),
                                                                         c2_df['ssim'].max()))
        print("mse")
        print('mean {0} vs {1} -> ssim {2} std -> {3} max -> {4}'.format(c_inv, c_inv - 1, np.mean(c1_df['mse']),
                                                                         np.std(np.array(c1_df['mse']), axis=0),
                                                                         c1_df['mse'].max()))
        print('mean {0} vs {1} -> ssim {2} std -> {3} max -> {4}'.format(c_inv, c_inv + 1, np.mean(c2_df['mse']),
                                                                         np.std(np.array(c2_df['mse']), axis=0),
                                                                         c2_df['mse'].max()))

    def save_data(self, name, data):
        with open(name, "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def load_data(self, file_name):
        try:
            with open(file_name, 'rb') as f:
                x = pickle.load(f)
        except:
            x = []
        return x

    def get_ssim_by_class_new(self, path, array_indices, array_images, row, col, flag_row = True):
        row_obj = pd.DataFrame(columns=['img_pivo', 'img', 'ssim', 'mse'])
        if flag_row:
            pivo = row
            row_row = row
            col_col = col
        else:
            pivo = col
            row_row = col
            col_col = row
        print(row, col)
        # os.system('pause')
        for pivo_class in array_indices[str(pivo)][str(pivo)]:
            for row_class in array_indices[str(row)][str(col)]:
                path_img_1 = os.path.join(path, 'images', array_images['path_img'].iloc[pivo_class].split('/')[1])
                path_img_2 = os.path.join(path, 'images', array_images['path_img'].iloc[row_class].split('/')[1])

                img1 = cv2.cvtColor(cv2.imread(path_img_1), cv2.COLOR_BGR2GRAY)
                img2 = cv2.cvtColor(cv2.imread(path_img_2), cv2.COLOR_BGR2GRAY)

                ssim_imgs = ssim(img1, img2)
                mse_imgs = self.mse(img1, img2)

                new = {'img_pivo': path_img_1, 'img': path_img_2, 'ssim': ssim_imgs, 'mse': mse_imgs}
                row_obj = row_obj.append(new, ignore_index=True)
        return row_obj

    def get_ssmi_all_classes_new(self, path, network, dataset_name):
        a = self.get_best_model('accuracy')

        df_images = pd.read_csv(os.path.join(path, 'gyro.txt'), sep=" ", engine="python", encoding="ISO-8859-1",
                                names=['path_img', 'real'])

        row_class_indices_matrix = {'0': {'0': [], '1': [], '2': [], '3': [], '4': []},
                            '1': {'0': [], '1': [], '2': [], '3': [], '4': []},
                            '2': {'0': [], '1': [], '2': [], '3': [], '4': []},
                            '3': {'0': [], '1': [], '2': [], '3': [], '4': []},
                            '4': {'0': [], '1': [], '2': [], '3': [], '4': []}
                            }

        df_exp = self._array_raw_exps[a['index']]

        for i in range(df_exp.shape[0]):
            row_class_indices_matrix[str(int(df_exp['real'][i]))][str(int(df_exp['pred'][i]))].append(i)

        tes = Test(self._num_classes)

        for row in range(num_classes):
            for col in range(num_classes):
                if row != col:
                    print(row, col)
                    tes._matrix_row_evaluate[row][col] = self.get_ssim_by_class_new(path, row_class_indices_matrix, df_images, row, col)
        print('=============')
        for col in range(num_classes):
            for row in range(num_classes):
                if row != col:
                    # print(col, row)
                    tes._matrix_col_evaluate[row][col] = self.get_ssim_by_class_new(path, row_class_indices_matrix, df_images, row, col, False)

        for i in range(self._num_classes):
            for j in range(self._num_classes):
                if i != j:
                    if not tes._matrix_row_evaluate[i][j].empty:
                        print(len(tes._matrix_row_evaluate[i][j]), end=' ')
                    else:
                        print(len(tes._matrix_row_evaluate[i][j]), end=' ')
                else:
                    print('pivo', end=' ')
            print()
        self.save_data(network + '_' + dataset_name + '_exp_' + str(self._exps[a['index']]) + '.pkl', tes)

    def print_similarity(self, metric, threshold, flag_threshold, exp, flag_prop):
        for i in range(self._num_classes):
            for j in range(self._num_classes):
                if i != j:
                    if flag_threshold:
                        if flag_prop == 'row':
                            filter = (exp._matrix_row_evaluate[i][j][metric] > threshold)
                            aux = exp._matrix_row_evaluate[i][j][metric][filter]
                        elif flag_prop == 'col':
                            filter = (exp._matrix_col_evaluate[i][j][metric] > threshold)
                            aux = exp._matrix_col_evaluate[i][j][metric][filter]
                    else:
                        if flag_prop == 'row':
                            aux = exp._matrix_row_evaluate[i][j][metric]
                        elif flag_prop == 'col':
                            aux = exp._matrix_col_evaluate[i][j][metric]
                    if metric == 'ssim':
                        if len(aux) > 0:
                            print("{mean:.5f}\xB1{std:.5f}".format(mean=np.mean(aux), std=np.std(aux)), end=' ')
                        else:
                            print('{mean:.5f}\xB1{std:.5f}'.format(mean=0, std=0), end=' ')
                    else:
                        if len(aux) > 0:
                            print("{mean:0>11.5f}\xB1{std:2.5f}".format(mean=np.mean(aux), std=np.std(aux)), end='  ')
                        else:
                            print('{mean:0>11.5f}\xB1{std:0>10.5f}'.format(mean=0, std=0), end='  ') 
                else:
                    if metric == 'ssim':
                        print('{:14}'.format('pivo'.rjust(10)), end='  ')
                    else:
                        print('{:22}'.format('pivo'.rjust(13)), end='  ')
            print()

    def generate_evaluate_similarity(self, obj_file_name):
        # print(len(exp._matrix_row_evaluate[0][1]['ssim']))
        threshold = 0.24
        flag_th = False
        print('==== row ssim =====')
        exp = self.load_data(obj_file_name)
        # plt.hist(exp._matrix_col_evaluate[1][0]['ssim'], 25, rwidth=0.4)
        # plt.boxplot(exp._matrix_col_evaluate[1][0]['ssim'])

        method_similarity = 'ssim'

        aux = (exp._matrix_col_evaluate[1][0][method_similarity] > 0)
        data_0 = exp._matrix_col_evaluate[1][0][method_similarity][aux]
        #print(len(data_0))

        data_1 = exp._matrix_col_evaluate[0][1][method_similarity]
        data_2 = exp._matrix_col_evaluate[2][1][method_similarity]
        data_3 = exp._matrix_col_evaluate[1][2][method_similarity]
        data_4 = exp._matrix_col_evaluate[3][2][method_similarity]
        data_5 = exp._matrix_col_evaluate[2][3][method_similarity]
        data_6 = exp._matrix_col_evaluate[3][4][method_similarity]

        data = [data_0, data_1, data_2, data_3, data_4, data_5, data_6]

        fig = plt.figure(figsize=(10, 7))
        # plt.xlabel("X-axis ")
        # Creating axes instance
        # ax = fig.add_axes([0, 0, 3, 4])
        ax = fig.add_subplot()

        # Creating plot
        # self.chart_test(data, ax)
        bp = ax.boxplot(data, vert=0)
        # print(bp['whiskers'][0].get_xdata(), 'asdf')
        # for medline in bp['whiskers']:
        #    linedata = medline.get_xdata()
        #    print(linedata)
        # os.system('pause')
        up_limit = []
        metric_aval = 'medians' #medians  whiskers
        up = False
        for i in range(len(bp[metric_aval])):
            if up:
                if i % 2 != 0:
                    up_limit.append(bp[metric_aval][i].get_xdata()[0])
            else:
                if i % 2 == 0:
                    up_limit.append(bp[metric_aval][i].get_xdata()[0])
        print('mean -> ', np.mean(np.array(up_limit)))
        #os.system('pause')


        plt.show()#official
        #os.system('pause')
        self.print_similarity('ssim', threshold, flag_th, exp, 'row')
        print('===== row mse ========')
        self.print_similarity('mse', threshold, flag_th, exp, 'row')
        print('====== column ssim ======')
        exp = self.load_data(obj_file_name)
        self.print_similarity('ssim', threshold, flag_th, exp, 'col')
        print('===== column mse ========')
        self.print_similarity('mse', threshold, flag_th, exp, 'col')

    def chart_test(self, data, ax):
        shape, loc, scale = stats.lognorm.fit(data, loc=1)
        pdf_lognorm = stats.lognorm.pdf(data, shape, loc, scale)

        #fig, ax = plt.subplots(figsize=(8, 4))

        ax.hist(data, bins='auto', density=True)
        #ax2 = ax.twinx()
        ax.plot(data, pdf_lognorm)
        #ax2.hist(data, bins='auto')
        ax.set_ylabel('probability')
        #ax2.set_ylabel('frequency')
        ax.set_title('Linear Scale')
        #plt.show()

    def get_ssim_by_class_dataset(self, path, dataset_name, clas, clas_comp):
        row_obj = pd.DataFrame(columns=['img_pivo', 'img', 'ssim', 'mse'])
        for i in range(len(clas)):
            for j in range(len(clas_comp)):
                path_img_1 = os.path.join(path, clas['phase'].iloc[i], dataset_name, 'images', clas['path_img'].iloc[i].split('/')[1])
                path_img_2 = os.path.join(path, clas_comp['phase'].iloc[j], dataset_name, 'images', clas_comp['path_img'].iloc[j].split('/')[1])

                # print(path_img_2)
                img1 = cv2.cvtColor(cv2.imread(path_img_1), cv2.COLOR_BGR2GRAY)
                img2 = cv2.cvtColor(cv2.imread(path_img_2), cv2.COLOR_BGR2GRAY)

                ssim_imgs = ssim(img1, img2)
                mse_imgs = self.mse(img1, img2)

                new = {'img_pivo': path_img_1, 'img': path_img_2, 'ssim': ssim_imgs, 'mse': mse_imgs}
                row_obj = row_obj.append(new, ignore_index=True)
        return row_obj

    def generate_evaluate_similarity_by_dataset(self, path_root, dataset_name):
        exps_types = ['test']#, 'val', 'train']
        all_images = pd.DataFrame(columns=['path_img', 'real'])
        for exp_type in exps_types:
            path = os.path.join(path_root, exp_type, dataset_name, 'gyro.txt')
            df_images = pd.read_csv(os.path.join(path), sep=" ", engine="python", encoding="ISO-8859-1",
                                    names=['path_img', 'real'])
            df_images['phase'] = exp_type
            all_images = all_images.append(df_images, sort=False)
        all_images = all_images.reset_index(drop=True)

        # print(all_images[(all_images['phase'] == 'val')])
        # os.system('pause')
        all_classes = []
        for i in range(self._num_classes):
            aux = (all_images['real'] == i)
            all_classes.append(all_images[aux])

        tes = Test(self._num_classes)

        prog_dim = np.zeros((self._num_classes, self._num_classes))
        for clas in range(self._num_classes):
            for clas_comp in range(self._num_classes):
                if clas != clas_comp and prog_dim[clas][clas_comp] != 1 and prog_dim[clas_comp][clas] != 1:
                    print(clas, clas_comp)
                    tes._matrix_row_evaluate[clas][clas_comp] = self.get_ssim_by_class_dataset(path_root, dataset_name, all_classes[clas], all_classes[clas_comp])
                    prog_dim[clas][clas_comp] = prog_dim[clas_comp][clas] = 1
        self.save_data(dataset_name + '.pkl', tes)


class Test(object):
    def __init__(self, num_classes):
        self._num_classes = num_classes
        self._matrix_row_evaluate = []
        self._matrix_col_evaluate = []
        self.init_config()

    def init_config(self):
        row_obj = pd.DataFrame(columns=['img_pivo', 'img', 'ssim', 'mse'])
        # for row in range(num_classes):
        #     row_evaluate.append([])
        for row in range(self._num_classes):
            row_array = []
            for col in range(self._num_classes):
                row_array.append(row_obj)
            self._matrix_row_evaluate.append(row_array)
        for row in range(self._num_classes):
            row_array = []
            for col in range(self._num_classes):
                row_array.append(row_obj)
            self._matrix_col_evaluate.append(row_array)
        # print(row_obj)

exps_path = "../../experiments"

# no transfer
no_vgg16_sidewalk_accy_158_pc_dataset = []
no_resnet50_sidewalk_accy_158_pc_dataset = []
no_dronet_sidewalk_accy_158_pc_dataset = []
no_squeezenet_sidewalk_accy_158_pc_dataset = []

# no transfer
no_vgg16_sidewalk_accy_flipped_315_pc_dataset = []
no_resnet_sidewalk_accy_flipped_315_pc_dataset = []
no_dronet_sidewalk_accy_flipped_315_pc_dataset = []
no_squeezenet_sidewalk_accy_flipped_315_pc_dataset = []

# transfer
vgg16_sidewalk_accy_flipped_315_pc_dataset = [260, 271, 272, 273, 274, 275, 276, 277, 278, 279]
resnet_sidewalk_accy_flipped_315_pc_dataset = [261, 262, 263, 264, 265, 266, 267, 268, 269, 270]
dronet_sidewalk_accy_flipped_315_pc_dataset = [250, 251, 252, 253, 254, 255, 256, 257, 258, 259]

# transfer learning
dronet_sidewalk_accx_184_pc_dataset = [134, 135, 136, 137, 138, 139, 140, 141, 142, 143]# , 144, 145, 146, 147, 148, 149, 150, 151, 152, 153]
resnet50_sidewalk_accx_184_pc_dataset= [154, 155, 156, 157, 158, 159, 160, 161, 162, 163]# , 164, 165, 166, 167, 168, 169, 170, 171, 172, 173]
vgg16_sidewalk_accx_184_pc_dataset = [127, 128, 129, 130, 174, 175, 176, 177, 178, 179] # , 180, 181, 182, 183, 184, 185, 186, 187, 188, 189]

# transfer learning
vgg16_indoor_accx_572_pc_dataset = [190, 191, 226, 227, 228, 229, 230, 231, 232, 233]
resnet50_indoor_accx_572_pc_dataset = [194, 195, 202, 203, 204, 205, 206, 207, 208, 209]
dronet_indoor_accx_572_pc_dataset = [198, 199, 234, 235, 236, 237, 238, 239, 240, 241]

# transfer learning
vgg16_indoor_accy_flipped_232_pc_dataset = [192, 193, 218, 219, 220, 221, 222, 223, 224, 225]
resnet50_indoor_accy_flipped_232_pc_dataset = [196, 197, 210, 211, 212, 213, 214, 215, 216, 217]
dronet_indoor_accy_flipped_232_pc_dataset = [200, 201, 242, 243, 244, 245, 246, 247, 248, 249]

# trasnfer learning with all datasets > 5k images
vgg16_sidewalk_all_datasets_velx = [280]
dronet_sidewalk_all_datasets_velx = [281]
resnet_sidewalk_all_datasets_velx = [282]

vgg16_sidewalk_all_datasets_vely  = [283]
resnet_sidewalk_all_datasets_vely = [284]
dronet_sidewalk_all_datasets_vely = [285]

# transfer learning with supermarket all dataset
# vgg16_market_all_datasets_velx = []
# resnet_market_all_datasets_velx = [287]
# dronet_market_all_datasets_velx = [288]
#
# vgg16_market_all_datasets_vely = [289]
# resnet_market_all_datasets_vely = [290]
# dronet_market_all_datasets_vely = [291]

# ======== new dataset sidewalk all accy ============
vgg16_sidewalk_accy_y_all_datasets_new_vely_1630 = [309, 310, 305, 311, 312, 313, 314, 315, 316, 317]
resnet_sidewalk_accy_y_all_datasets_new_vely_1630 = [318, 319, 320, 321, 322, 323, 324, 325, 326, 327]
dronet_sidewalk_accy_y_all_datasets_new_vely_1630 = [335]# [328, 329, 330, 331, 332, 333, 334, 335, 336, 337]
vgg16_sidewalk_accy_y_all_datasets_new_vely_1222 = [303]
vgg16_sidewalk_accy_y_all_datasets_new_vely_978 = [302]

# ======== new dataset sidewalk  ============
vgg16_sidewalk_accx_x_all_datasets_new_velx_4023_4_1 = [308, 338, 339, 340, 341, 342, 343, 344, 345, 346]
resnet50_sidewalk_accx_x_all_datasets_new_velx_4023_4_1 = [357, 359, 360, 361, 362, 363, 364, 365, 366]
dronet_sidewalk_accx_x_all_datasets_new_velx_4023_4_1 = [347, 348, 349, 350, 351, 352, 353, 354, 355, 356]
vgg16_sidewalk_accx_x_all_datasets_new_velx_3576 = [300]
vgg16_sidewalk_accx_x_all_datasets_new_velx_2682 = [301]
vgg16_sidewalk_accx_x_all_datasets_new_velx_2145 = [299]
vgg16_sidewalk_accx_x_all_datasets_new_velx_4290 = [291]
vgg16_sidewalk_accx_x_all_datasets_new_velx_4470 = [306]
vgg16_sidewalk_accx_x_all_datasets_new_velx_4470_12_2 = [307]
vgg16_sidewalk_accx_x_all_datasets_new_velx_4291 = [293]
vgg16_sidewalk_accx_x_all_datasets_new_velx_4827 = [294]
vgg16_sidewalk_accx_x_all_datasets_new_velx_5345 = [295]
vgg16_sidewalk_accx_x_all_datasets_new_velx_3218 = [297]

# ======== new market dataset all accx ==============
resnet50_market_accx_x_all_datasets_new_velx_1414 = [367, 368, 369, 370, 371, 372, 373, 374, 375, 376]
vgg16_market_accx_x_all_datasets_new_velx_1414 = [377, 378, 379, 380, 381, 382, 383, 384, 385, 386]
dronet_market_accx_x_all_datasets_new_velx_1414 = [387, 388, 389, 390, 391, 392, 393, 394, 395, 396]

# ======== new market dataset all accy ==============
resnet50_market_accy_y_all_datasets_new_velx_362 = [397, 398, 399, 400, 401, 402, 403, 404, 405, 406]
vgg16_market_accy_y_all_datasets_new_velx_362 = [407]#, 408, 409, 410, 411, 412, 413, 414, 415, 416]
dronet_market_accy_y_all_datasets_new_velx_362 = [417]#, 418, 419, 420, 421, 422, 423, 424, 425, 426]

vgg16_market_accx_x_all_datasets_ssim_new_velx_1395 = [427]
vgg16_market_accx_x_all_datasets_ssim_new_velx_1386 = [428, 430, 431, 432, 433, 434, 435, 436, 437, 438]
vgg16_market_accx_x_all_datasets_mse_new_velx_1380 = [429]
resnet50_market_accx_x_all_datasets_ssim_new_velx_1386 = [439]#, 440, 441, 442, 443, 444, 445, 446, 447, 448]
dronet_market_accx_x_all_datasets_ssim_new_velx_1386 = [449, 450, 451, 452, 453, 454, 455, 456, 457, 458]

## ------ novos testes ---------

dronet_market_2_x_proportional_all = [459,460,461,462,463]
dronet_market_2_x_proportional_all_transfer_with_last_model = [464, 465, 466, 467, 468]
market_dataset_2_y_proportional_flipped_classes_all_transfer_with_last_model = [469, 470, 471, 472, 473]
market_x_2_regress = [479]

#iii teste com dataset sidewalk com cropped
sidewalk_accy_all_datasets_classes_new_1630_cropped = [474, 475, 476, 477, 478]

set = "test"
exps_phase = "test"
name_weights = "model_weights_299.h5" # dronet 299
#name_weights = "model_weights_99.h5" # dronet 299
num_classes = 5

res = ResultsExperiments(exps_path, market_x_2_regress, exps_phase, name_weights, set, num_classes, load_bests=False, percent=True)
# res.get_results_log(vgg16_sidewalk_all_datasets_vely ) # Get results to print.
# res.print_best_result() # Print metrics log of experiments
# res.print_best_exps(metric="val_loss") # Print best model each exp
res.generate_statistics()

network = 'vgg16'
dataset_name = 'sidewalk_accx_all_datasets_classes_new_4023_4_1_03'
path_root = os.path.join('../../datasets', 'vgg16', dataset_name, dataset_name)

# path = os.path.join('./../../datasets/vgg16/sidewalk_accx_all_datasets_classes_new_4023_4_1_00/sidewalk_accx_all_datasets_classes_new_4023_4_1_00/sidewalk_accx_all_datasets_classes_new_4023_4_1_00/test/sidewalk_accx_all_datasets_classes_new_4023_4_1_00/gyro.txt')
# res.get_ssmi_by_classes(path) # obsoleto

# res.get_ssmi_all_classes_new(os.path.join(path_root, 'test', dataset_name), network, dataset_name) # generate file obj
exp_name = 'exp_378'
# file_name = network + '_' + dataset_name + '_' + exp_name + '.pkl'
file_name = 'vgg16_market_accx_all_datasets_classes_1414_01_exp_378.pkl'
# res.generate_evaluate_similarity(file_name) # load evaluate and make statistics

# res.generate_evaluate_similarity_by_dataset(path_root, dataset_name)
