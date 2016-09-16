""""This class is responsible for loading the data from disk, as well as handling data feeding to the optimization
algorithm, based on epochs.
Data can be shuffled. Class can return group scores, instance scores etc"""
import random
import numpy as np
import time

"""
Data files must have the following format:
One group of instances per line, and score of group at the end of the line.
Instances are split with '\t'
Features withtin instances are split with ' ' spaces.
different separators must be explicitly given in load_data_embeddings.
After the last \t is the score of the group either 1 or 0
Its called embeddings file, because an embedding are used for each instance
"""
MIN_INSTANCES = 10


class DataHandler(object):
    def __init__(self, embeddings_file, max_size=200000,  quiet=False):
        if not quiet:
            print 'loading data...',
        start = time.time()
        self.embeddings_file = embeddings_file  # file with data . One group of data at each line, and score at the end
        self.max_size = max_size  # max number of groups
        self.current_group_index = 0
        self.current_instance_index = 0
        self.batch_size = 0
        self.data_dictionary = {}
        self.negatives = 0
        self.positives = 0
        self.instances, self.group_labels, self.group_lengths = self.load_data_embeddings(embeddings_file)

        if not quiet:
            print 'Total instances: ', len(self.instances), ' in ', len(self.group_labels), ' groups',
            print 'Positives: ', self.positives, ' Negatives: ', self.negatives
            duration = time.time() - start
            print 'done  in', duration, 'seconds'

    def load_data_embeddings(self, embeddings_file, instance_sep='\t', feature_sep=' '):  # each line is a group
        """Loads the file with the embeddings. Each line corresponds to one group. Each embedding corresponds to one
        instance. Also counts number of positive and negative groups"""
        instances = []
        group_scores = []
        group_lengths = []  # needed to find limits of groups

        d = 1
        with open(embeddings_file) as f:
            for line in f:
                if d % 10000 == 0:
                    print d, '... ',
                if len(line) < MIN_INSTANCES:
                    continue  # remove groups with less than MIN_INSTANCES instances

                elements = line.split(instance_sep)
                score = float(elements[len(elements) - 1].strip())

                group = []
                for i in range(len(elements) - 1):
                    features = elements[i].split(feature_sep)
                    emb = [float(num) for num in features[0:len(features) - 1]]
                    instances.append(emb)
                    group.append(emb)

                group_scores.append(score)
                group_lengths.append(len(elements) - 1)
                if score == 0:  # for stats
                    self.negatives += 1
                else:
                    self.positives += 1

                self.data_dictionary[d] = (group, score)
                d += 1
                if d >= self.max_size:
                    break

            X = np.array(instances, dtype='float16')
            group_scores = np.array(group_scores, dtype='uint8')
            group_lengths = np.array(group_lengths, dtype='uint16')
            del instances  # memory save
            return X, group_scores, group_lengths

    def _shuffle_data(self):
        """Shuffles the data randomly. A bit more complicated because has to maintain order of group lengths the same
        way"""
        X = np.zeros(self.instances.shape)
        Y = np.zeros(self.group_labels.size, dtype='uint8')
        new_len = np.zeros(self.group_labels.size, dtype='uint16')
        indices = np.arange(len(Y))  # 1 2 3 4...
        random.shuffle(indices)  # re-arragned

        frm = 0
        for i in range(len(indices)):
            Y[i] = self.group_labels[indices[i]]
            new_len[i] = self.group_lengths[indices[i]]
            to = frm + new_len[i]
            frm2 = np.sum(self.group_lengths[0:indices[i]])
            to2 = frm2 + self.group_lengths[indices[i]]

            X[frm:to, ] = self.instances[frm2:to2, ]
            frm += new_len[i]

        self.group_lengths = None
        self.group_labels = None
        self.instances = None
        self.group_lengths = new_len
        self.group_labels = Y
        self.instances = X

        del X
        del Y
        del new_len

    def get_next_batch(self):
        """Returns the next batch of data, based on the number of groups in mini-batch size.
        Has to calculate the number of instances that belong there based on group lengths"""
        if self.current_group_index >= len(self.group_lengths):  # epoch ended
            return None, None, None

        frm = self.current_group_index
        to = frm + self.batch_size
        Y = self.group_labels[frm:to]
        lengths = self.group_lengths[frm:to]
        length = np.sum(lengths)
        frm = self.current_instance_index
        to = frm + length
        X = self.instances[frm:to, ]

        self.current_group_index += self.batch_size
        self.current_instance_index += length

        return X, Y, lengths

    def rewind_dataset(self, shuffle=True):
        """Shuffle dataset if necessary and restart mini-batch process"""
        if shuffle:
            self._shuffle_data()
        self.current_instance_index = 0
        self.current_group_index = 0

    # getters and setting batch size
    def get_instances(self):
        return self.instances

    def get_group_labels(self):
        return self.group_labels

    def get_lengths(self):
        return self.group_lengths

    def get_embeddings_dimension(self):
        return self.instances.shape[1]

    def set_batch_size(self, bs):
        self.batch_size = bs
