import os
import pandas as pd
import numpy as np

class PredictedDatasets:
    def __init__(self, path, fileName):
        self.path = path
        self.fileName = fileName
        self.results = None
        self.loadResults()
        pass

    def loadResults(self):
        fileLoaded = os.path.join(self.path, self.fileName)
        self.results = pd.read_csv(fileLoaded, sep=" ", engine="python", encoding="ISO-8859-1", names=['pred', 'real'])
        self.results['index'] = self.results.index
        # print(self.results)
        # os.system("pause")

class SuiteStatistics:
    def __init__(self, path, fileNameNormal="", fileNameLite=""):
        self.path = path
        self.normal = PredictedDatasets(path, fileNameNormal)
        self.lite = PredictedDatasets(path, fileNameLite)
        pass

    def getPercentageErrosLite(self, classesQtd=5):
        # print(self.normal.results)
        self.matrixErrosLite = []
        self.matrixHitsLite = []
        self.normal.results.astype(int)
        self.lite.results.astype(int)
        for classe in range(classesQtd):
            normalErros = (self.normal.results['pred'] != classe) & (self.normal.results['real'] == classe)
            liteErros = (self.lite.results['pred'] != classe) & (self.lite.results['real'] == classe)
            normalHits = (self.normal.results['pred'] == classe) & (self.normal.results['real'] == classe)
            liteHits = (self.lite.results['pred'] == classe) & (self.lite.results['real'] == classe)

            #liteErrorsIndex = self.lite.results.index[liteErros]
            #normalErrosIndex = self.normal.results.index[normalErros]


            #print(self.lite.results[liteErros])
            #print(self.normal.results[normalErros])

            #a = self.lite.results[liteErros].copy()
            #b = self.normal.results[normalErros].copy()

            #a['index'] = a.index
            #b['index'] = b.index

            #print(a)
            #print(b)

            c = self.lite.results[liteErros].merge(self.normal.results[normalErros], how='inner', indicator=False)
            #print(c)

            #print(self.lite.results[liteErros].drop(c['index'])) #aqui o lite divergiu
            self.matrixErrosLite.append(self.extractErros(self.lite.results[liteErros].drop(c['index'])))

            d = self.lite.results[liteHits].merge(self.normal.results[normalHits], how='inner', indicator=False)

            self.matrixHitsLite .append(self.extractErros(self.lite.results[liteHits].drop(d['index'])))
        # print('-------- erros -----')
        # for i in range(5):
        #     print(self.matrixErrosLite[i])
        #
        # for i in range(5):
        #     print(self.matrixHitsLite[i])

        m_erros = np.array(self.matrixErrosLite)
        m_hits = np.array(self.matrixHitsLite)
        print('-------- erros -----')
        print(m_erros)
        print('------- hits ------')
        print(m_hits)
        #os.system('pause')
        m_total = m_erros + m_hits
        print('----- total -----')
        print(m_total)
        print('---- percent ----')
        print("Divergences {0}".format(m_total.sum()))
        print("Percentage {0}".format(m_total.sum()/self.lite.results.shape[0]))

    def extractErros(self, liteErros):
        classes = np.zeros(5)
        for i, row in liteErros.iterrows():
        #for classe in liteErros.values:
            classes[int(row['pred'])] += 1
        #print(classes)
        return classes

    def getStatistics(self):
        self.getPercentageErrosLite()




path = "../../experiments/exp_335"
normal = SuiteStatistics(path, "predict_truth_test_model_weights_299.h5_0_normal_sidewalk_y.txt",
                         "predict_truth_test_model_weights_299.h5_0_lite_sidewalk_y.txt")
normal.getStatistics()




