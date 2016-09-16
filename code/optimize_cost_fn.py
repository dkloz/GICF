"""Contains the class that implements the main part of the paper.
Loads the data, Sets the parameters and then optimizes the cost function until we run out of epochs or some termination
conditions are met.
"""
import sentence_similarity as similarity
import numpy as np
import array_functions as af
import util.io as io
from util.data_handler import DataHandler
from evaluation import GroupEvaluator, InstanceEvaluator

np.random.seed(12345)  # in order for data to be shuffled randomly, but runs to be identical


class GICF(object):
    """Defines the model described in the paper. Creates filenames for each dataset and loads the data.
    Then optimizes the cost function through the train method"""

    def __init__(self, dataset='movies'):
        """Initialize variables, Get filenames according to dataset load data and set parameters"""
        self.dataset = dataset
        self.filenames = self.get_filenames(dataset)
        self.test_groups = []
        self.test_instances = []
        self.train_data = None
        self.train_eval = None
        self.test_eval = None
        self.instance_eval = None
        self.group_acc = []
        self.instance_auc = []
        self.instance_acc = []
        self.train_acc = []
        self.total_iterations = 0

        self.load_data()  # takes a few seconds
        self.set_parameters()  # set default parameters. Can be changed later
        self.embeddings_dimension = self.train_data.get_embeddings_dimension()
        self._print_titles = '#iter\tACC Train\tAUC Train\tACC Test\tAUC Test\t\t|\tACC Sent\tAUC Sent\t\tPRC Sent'

    def get_filenames(self, dataset='movies'):
        """Sets the filenames based on each dataset"""
        if dataset == 'movies':
            dir_name = './data/movies/'
        elif dataset == 'yelp':
            dir_name = 'data/yelp/'
        elif dataset == 'amazon':
            dir_name = 'data/amazon/cells/'
        else:
            print 'Wrong dataset Name.'
            return

        embeddings_file_train = dir_name + 'train.emb'  # emb\t emb\t score\n
        embeddings_file_test = dir_name + 'test.emb'
        instance_labels = dir_name + 'test_sentences.emb'
        return embeddings_file_train, embeddings_file_test, instance_labels

    def load_data(self):
        """Makes the data handlers, and loads the data from disk"""
        emb_train, emb_test, instance_labels = self.filenames
        self.train_data = DataHandler(emb_train, max_size=200000)
        self.train_eval = GroupEvaluator(data=self.train_data)
        self.test_eval = GroupEvaluator(data=DataHandler(emb_test, max_size=200000))
        self.instance_eval = InstanceEvaluator()
        self.instance_eval.load_labeled_instances(instance_labels)

    @property
    def _param_str(self):
        """A string with the parameters of the experiment"""
        return str(self.epochs) + 'x' + str(self.batch_size) + '_' + str(self.lr) + '_' + str(
            self.alpha_balance) + self.similarity_fn + str(self.sim_variance)

    def set_parameters(self, batch_size=50, alpha_balance=0.04, lr=0.05, momentum_value=0.7, similarity_fn='rbf',
                       sim_variance=0.7071, epochs=3):
        """Set the parameters for the run/experiment"""
        self.alpha_balance = alpha_balance
        self.momentum_value = momentum_value
        self.similarity_fn = similarity_fn
        self.sim_variance = sim_variance
        self.epochs = epochs
        self.batch_size = batch_size

        self.lr = lr * self.batch_size  # learning rate is a funciton of batch size
        self.run_name = self._param_str
        self.dir_name = self.similarity_fn + '_' + str(self.batch_size) + '_' + str(self.epochs)
        self.output_name = './training_output/' + self.dataset + '/' + self.dir_name + '_'
        self.train_data.set_batch_size(batch_size)

    def train(self):
        """Where the magic happens. Optimizes the cost function of the paper, based on the parameters given before.
        There is a terminating function which determines if optimization should end before the epochs end,
        based on essentially heuristics. Every 50 iterations prints progress. Keeps the best theta values based on the
        group reconstruction score. At the end prints detailed stats about classifying with that."""
        print 'Optimizing for ', self._param_str
        self.total_iterations = 0
        accs = []
        theta = np.random.random(self.embeddings_dimension)
        best_theta = theta
        best_acc = 0
        terminate = False

        for epoch in range(self.epochs):
            self.train_data.rewind_dataset(True)  # reset and shuffle data

            if terminate:
                break
            print '-------epoch ', epoch, '-----------'
            print self._print_titles

            X, gs, gl = self.train_data.get_next_batch()

            while X is not None:  # for each mini-batch # do gd step

                W_ij = similarity.get_sim_matrix(X, self.similarity_fn, self.sim_variance)

                # calculate y_hat and derivative
                Y_ij = af.calculate_y(X, theta)
                Y_der_ij = af.calculate_y_der(Y_ij, X)

                # calculate cost
                similarity_cost = af.similarity_derivative(Y_ij, Y_der_ij, W_ij) / (X.shape[0] ** 2)
                group_cost = self.alpha_balance * af.group_derivative(Y_ij, Y_der_ij, gs, gl) / float(len(gs))
                theta_der = similarity_cost + group_cost

                # new theta
                theta = self.momentum_value * theta - (1 - self.momentum_value) * self.lr / (epoch + 1) * theta_der
                self.total_iterations += 1

                # print progress
                if self.total_iterations % 50 == 0:
                    acc = self._print_progress(theta)
                    accs.append(acc)
                    if acc > best_acc:  # save best theta, based on training set
                        best_acc = acc
                        best_theta = theta
                        io.save_theta(theta, self.output_name + self._param_str, best=True)

                    if self._terminate_conditions(theta, accs):
                        terminate = True
                        break
                X, gs, gl = self.train_data.get_next_batch()

        io.save_theta(theta, self.output_name + self._param_str + '_last')

        print '\n\n\n\t\t\t---BEST THETA VALUE (in training group)---'

        self._print_progress(best_theta, print_details=True)
        return self.train_acc, self.group_acc, self.instance_acc, self.instance_auc

    def _terminate_conditions(self, theta, accs):
        if np.isnan(theta[0]):
            return True

        variance = np.array(accs)
        if len(variance) > 50:
            variance = variance[:-50]  # last 50 values
        var = np.var(variance)

        if self.total_iterations > 1500 and var < 0.00005:
            return True

    def _print_progress(self, theta, print_details=False):
        # iterations  train accuracy, train AUC, test accuracy, test AUC | instance accuracy, instance auc, instance PRC
        print '%6d\t' % self.total_iterations,
        acc_train, auc_train = self.train_eval.evaluate_groups(theta, print_details)

        self.train_acc.append([acc_train])
        print round(100 * acc_train, 2), ' \t\t(', round(100 * auc_train, 2), ')\t',
        acc, auc = self.test_eval.evaluate_groups(theta, print_details)
        self.group_acc.append(acc)
        print round(100 * acc, 2), ' \t\t(', round(100 * auc, 2), ')\t',
        print '\t|\t',
        acc, auc = self.instance_eval.evaluate_instances(theta)
        auprc = self.instance_eval.evaluate_instances(theta, prc=True)

        self.instance_acc.append(acc)
        self.instance_auc.append(auc)

        print round(100 * acc, 2), '\t\t(', round(100 * auc, 2), ')\t', ' \t(', round(100 * auprc, 2), ')\t'
        return acc_train  # based on this we decide best theta
