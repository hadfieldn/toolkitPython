from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .matrix import Matrix


class BaselineLearner(SupervisedLearner):
    """
    For nominal labels, this model simply returns the majority class. For
    continuous labels, it returns the mean value.
    If the learning model you're using doesn't do as well as this one,
    it's time to find a new learning model.
    """

    labels = []

    def __init__(self):
        pass

    def train(self, features, labels):
        """
        :type features: Matrix
        :type labels: Matrix
        """
        self.labels = []
        for i in range(labels.cols):
            if labels.value_count(i) == 0:
                self.labels += [labels.column_mean(i)]          # continuous
            else:
                self.labels += [labels.most_common_value(i)]    # nominal

    def predict(self, features, labels):
        """
        :type features: [float]
        :type labels: [float]
        """
        del labels[:]
        labels += self.labels



