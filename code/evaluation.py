"""Class that deals with the evaluation part of the process. Contains evaluation for groups and instances
Given some value of theta, """
import numpy as np
import array_functions as af
from result_analysis import ResultAnalyzer
import util.io as io


class GroupEvaluator(object):
    """A class that helps evaluate classification on the group level."""
    def __init__(self, filename=None, theta=None, data=None):
        """Initialize with filename for parameters theta or actual numpy array. These are helpful for evaluation after
        training."""
        if filename is not None:  # Can provide filename with value of theta
            self.theta = io.load_theta(filename)
        else:
            self.theta = theta
        self.data = data
        self.labeled_data = False

    # Evaluates the loaded data, against its self. on a group level
    def evaluate_groups(self, theta=None, print_details=False):
        """Calculate group scores and pass them in result analyzer and return proper metric"""
        if theta is not None:
            self.theta = theta
        ra = ResultAnalyzer()
        y_hat = self.get_group_scores()  # get the score for each group
        ra.addManyResults(self.data.get_group_labels(), y_hat)
        acc = ra.accuracy()
        auc = ra.auc()
        if print_details:
            ra.print_details()
        return acc, auc

    def get_group_scores(self):
        """Gets the score for each instance, and then averages them out. It knows number of instances in each group
        but group length"""
        gl = self.data.get_lengths()
        scores = np.zeros(len(gl))
        y_hat = self.get_instance_scores()
        frm = 0
        for i in range(len(gl)):
            if gl[i] == 0:
                scores[i] = 0
                continue
            to = frm + gl[i]
            scores[i] = np.average(y_hat[frm:to])
            frm = to
        return scores

    def get_instance_scores(self):
        x = self.data.get_instances()
        return af.calculate_y(x, self.theta)


class InstanceEvaluator(object):
    """Similar class to the group evaluator, but it works for evaluating how well we do on instances. Each instance
    has one label. """
    def __init__(self, filename=None, theta=None, data=None):
        if filename is not None:
            self.theta = io.load_theta(filename)
        else:
            self.theta = theta
        self.data = data
        self.labeled_data = True
        self.eval_data = None
        self.y_known = None

    def load_labeled_instances(self, filename, separator=' ', quiet=False):
        """This is for supervised evaluation: data must be in format:
            instance \t score
            instance = feature [space] feature"""
        pos = 0
        neg = 0
        self.labeled_data = True
        y_known = []
        eval_data = []
        with open(filename) as f:
            for line in f:
                elements = line.split('\t')
                score = float(elements[len(elements) - 1])
                if score == 0:
                    neg += 1
                else:
                    pos += 1
                features = np.fromstring(elements[0], dtype=float, sep=separator)
                y_known.append(score)
                eval_data.append(features)

            self.eval_data = np.array(eval_data)
            self.y_known = np.array(y_known)
        if not quiet:
            print 'Instances:  positive: ', pos, ' Negative: ', neg

    def evaluate_instances(self, theta=None, prc=False):
        """Pass the instance scores and ground truth to result analyzer and return proper metric"""
        if not self.labeled_data:
            print 'I have no instance data. Must set them, and labels first'
            return None
        if theta is not None:   # theta overrides existing one if given
            self.theta = theta
        ra = ResultAnalyzer()
        y_hat = af.calculate_y(self.eval_data, self.theta)
        ra.addManyResults(self.y_known, y_hat)
        if prc:
            return ra.auprc()
        else:
            return ra.accuracy(), ra.auc()
