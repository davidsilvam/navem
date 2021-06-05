import os
import pandas as pd
import numpy as np

class ResultsExperiments():
    def __init__(self, exps_path, exps, phase, name_weigths, set, num_classes, load_bests=False):
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

    def generate_statistics(self):
        """
        Generate all results of statistics and get best model
        :return:
        """
        self.get_metrics()
        mean_accuracy = self.mean_accuracy(self._all_accuracy)
        mean_precision = self.mean_precision(self._all_precision)
        print("======== Metrics ===========")
        print("metric, mean, std")
        print("accuracy, {0}, {1}".format(mean_accuracy[0], mean_accuracy[1]))
        print("precision, {0}, {1}".format(mean_precision[0], mean_precision[1]))
        print("recall, {0}, {1}".format(self.mean_recall(self._all_recall)[0], self.mean_recall(self._all_recall)[1]))
        print("f1_score, {0}, {1}".format(self.mean_f1_score(self._all_F1_score)[0], self.mean_f1_score(self._all_F1_score)[1]))

        best_model = self.get_best_model("accuracy")
        print("======= Best model =========")
        print("Exp {0}".format(self._exps[best_model["index"]]))
        print("------ Accuracy -------")
        print("Accuracy -> {0}".format(best_model["max_value"]))
        print(self._all_confusion_matrix[best_model["index"]])
        print("------ Precision -------")
        self.print_metric_by_class(self._all_precision[best_model["index"]])
        print("Mean precision -> {0}".format(np.mean(self._all_precision[best_model["index"]])))
        print("------ Recall -------")
        self.print_metric_by_class(self._all_recall[best_model["index"]])
        print("Mean recall -> {0}".format(np.mean(self._all_recall[best_model["index"]])))
        print("------ F1 score ------")
        self.print_metric_by_class(self._all_F1_score[best_model["index"]])
        print("Mean F1 score -> {0}".format(np.mean(self._all_F1_score[best_model["index"]])))

    def print_metric_by_class(self, matrix):
        """
        Print results per class
        :param matrix: Confusion matrix
        :return:
        """
        for value, clas in enumerate(matrix):
            print("Class {0} -> {1}".format(value, clas))

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

set = "test"
exps_phase = "test"
name_weights = "model_weights_99.h5" # dronet 299
num_classes = 5

res = ResultsExperiments(exps_path, resnet50_sidewalk_accx_184_pc_dataset , exps_phase, name_weights, set, num_classes, load_bests=False)
res.get_results_log(resnet50_sidewalk_accx_184_pc_dataset ) # Get results to print.
# res.print_best_result() # Print metrics log of experiments
# res.print_best_exps(metric="val_loss") # Print best model each exp
res.generate_statistics()
