import os
import pandas as pd
import numpy as np

class ResultsExperiments():
    def __init__(self, exps_path, exps, phase, name_weigths, set, num_classes):
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

    def load_results_file(self):
        """
        Load file from results of experiments
        :return:
        """
        for exp in self._exps:
            name = ""
            if self._phase == "train":
                name = os.path.join(self._exps_path, "exp_" + str(exp),
                                      "predict_truth_" + set + "_" + self._name_weights + "_1_" + ".txt")
            else:
                name = os.path.join(self._exps_path, "exp_" + str(exp),
                                      "predict_truth_" + set + "_" + self._name_weights + "_0_" + ".txt")
            df = pd.read_csv(name, sep=" ", engine="python", encoding="ISO-8859-1", names=['pred', 'real'])
            self._array_raw_exps.append(df)

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
        for clas in range(len(matrix)):
            f1_score_classes.append((2 * self.precision_class(clas, matrix) * self.recall_class(clas, matrix)) / (self.precision_class(clas, matrix) + self.recall_class(clas, matrix)))
        return f1_score_classes

    def f1_score(self):
        """
        Obtain F1 score from whole experiments
        :return:
        """
        for matrix in self._all_confusion_matrix:
            self._all_F1_score.append(self.get_f1_score(matrix))

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
        print("Exp {0}".format(best_model["index"]))
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
exps_dronet_velx = [134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153]
set = "test"
exps_phase = "test"
name_weights = "model_weights_299.h5"
num_classes = 5

res = ResultsExperiments(exps_path, exps_dronet_velx, exps_phase, name_weights, set, num_classes)
res.generate_statistics()



