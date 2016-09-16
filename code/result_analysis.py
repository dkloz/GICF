"""A file that calculates detailed metrics on the outcome. Eg accuracy, AUC, PRC etc, using scikit learn"""
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn import metrics


class ResultAnalyzer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.size = 0
        self.positives = []
        self.negatives = []
        self.y = []
        self.y_hat = []

    def get_tp_fp_rates(self, y, y_hat):
        tpr, fpr, _ = metrics.roc_curve(y, y_hat, pos_label=1)
        return tpr, fpr

    # tp = True Positive => Guessed True, real was positive
    def addResult(self, score, true_sentiment):
        self.y.append(true_sentiment)
        self.y_hat.append(score)
        agree = False

        self.size += 1
        if true_sentiment > 0.5:
            self.positives.append(score)
            if score > 0.5:
                self.tp += 1
                agree = True
            else:
                self.fp += 1
        else:
            self.negatives.append(score)
            if score <= 0.5:
                self.tn += 1
                agree = True
            else:
                self.fn += 1
        return agree

    def accuracy(self):
        return float(self.tp + self.tn) / self.size

    def precision(self):
        offset = 0
        if (self.tp + self.fp) == 0:
            offset = 1
        return float(self.tp) / (self.tp + self.fp + offset)

    def recall(self):
        return float(self.tp) / max(1, (self.tp + self.fn))

    def f1(self):
        return float(2 * self.tp) / (2 * self.tp + self.fp + self.fn)

    def positiveAverage(self):
        return np.average(np.array(self.positives))

    def negativeAverage(self):
        return np.average(np.array(self.negatives))

    def getDeviation(self, sentiment='positive'):
        if sentiment is 'positive':
            return np.std(np.array(self.positives))
        else:
            return np.std(np.array(self.negatives))

    def addManyResults(self, Y, Y_hat):

        for i in range(len(Y)):
            self.addResult(Y_hat[i], Y[i])
        return

    # returns acciracy, precision, true positive, true negative, false positive, false negative
    def get_all_results(self):
        return self.accuracy(), self.precision(), self.tp, self.tn, self.fp, self.fn

    def get_AUC_score(self, Y_hat, Y, pos_limit=1):
        pos = []
        neg = []
        i = 0
        for y in Y:
            if y >= pos_limit:
                pos.append(Y_hat[i])
            else:
                neg.append(Y_hat[i])
            i += 1

        sum = 0.0
        for p in pos:
            for n in neg:
                if (p > n):
                    sum += 1

        sum = sum / (len(pos) * len(neg))
        return sum

    def auprc(self):
        precision, recall, thresholds = precision_recall_curve(self.y, self.y_hat)
        area = auc(recall, precision)
        return area

    def auc(self, average='macro'):

        y_hat = np.zeros(len(self.y_hat))
        for i in range(len(self.y_hat)):
            if (self.y_hat[i] > 0.5):
                y_hat[i] = 1
        try:
            auc = metrics.roc_auc_score(y_hat, self.y, average=average)
            return auc
        except:

            return 0

    def get_auc(self, y, y_hat):
        return metrics.roc_auc_score(y_hat, y, average='macro')

    def get_acc(self, y, y_hat):
        a = np.array(y)
        b = np.array(y_hat)
        return np.sum(a == b) / len(y)

    def print_details(self):
        print
        total_pos = self.tp + self.fp
        total_neg = self.tn + self.fn
        print '\tTrue\tFalse\t\tTotal'
        print 'Pos', self.tp, '\t', self.fp, ' \t|\t\t', total_pos
        print 'Neg', self.tn, '\t', self.fn, ' \t|\t\t', total_neg
        print '\t', str(self.tp + self.tn), '\t', str(self.fp + self.fn), '\t\t\t|', str(total_pos + total_neg)
        print '------------------------------------'

        print 'Accuracy', self.accuracy()
        print 'Precision', self.precision()
        print 'Recall', self.recall()
        print 'AUC: ', self.auc(), ' ', self.auc(average='micro')

        print '\nPositives',
        print 'Average: ', self.positiveAverage(), '\tDeviation: ', self.getDeviation('positive'), '\ttotal', len(
            self.positives)
        print '\nNegatives',
        print 'Average: ', self.negativeAverage(), '\tDeviation: ', self.getDeviation('negative'), '\ttotal', len(
            self.negatives)
