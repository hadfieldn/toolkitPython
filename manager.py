from supervised_learner import SupervisedLearner
from baseline_learner import BaselineLearner
from matrix import Matrix
import random
import argparse
import textwrap
import time


class MLSystemManager:

    def get_learner(self, model):
        """
        :type model: str
        :rtype: SupervisedLearner
        """

        # When you make a new learning algorithm, you should add al ine for it to this method.

        if model == "baseline":
            return BaselineLearner()
        # elif model == "perceptron":
        #     return Perceptron()
        # elif model == "neuralnet":
        #     return NeuralNet()
        # elif model == "decisiontree":
        #     return DecisionTree()
        # elif model == "knn"
        #     return InstanceBasedLearner()
        else:
            raise Exception("Unrecognized model: {}".format(model))

    def main(self):
        random.seed(0)      # Use a seed for deterministic results (makes debugging easier)

        # parse the command-line arguments
        args = self.parser().parse_args()
        file_name = args.arff
        learner_name = args.L
        eval_method = args.E[0]
        eval_parameter = args.E[1] if len(args.E) > 1 else None
        print_confusion_matrix = args.verbose
        normalize = args.normalize

        # load the model
        learner = self.get_learner(learner_name)

        # load the ARFF file
        data = Matrix()
        data.load_arff(file_name)
        if normalize:
            print("Using normalized data")
            data.normalize()

        # print some stats
        print(textwrap.dedent("""\

            Dataset name: {}
            Number of instances: {}
            Number of attributes: {}
            Learning algorithm: {}
            Evaluation method: {}

            """).format(file_name, data.rows, data.cols, learner_name, eval_method))

        if eval_method == "training":

            print("Calculating accuracy on training set...")

            features = Matrix(data, 0, 0, data.rows, data.cols)
            labels = Matrix(data, 0, data.cols-1, data.rows, 1)
            confusion = Matrix()

            start_time = time.time()
            learner.train(features, labels)
            elapsed_time = time.time() - start_time
            print("Time to train (in seconds): {}".format(elapsed_time))

            accuracy = learner.measure_accuracy(features, labels, confusion)
            print("Training set accuracy: " + accuracy)

            if print_confusion_matrix:
                print("\nConfusion matrix: (Row=target value, Col=predicted value")
                confusion.print()
                print("")

        elif eval_method == "static":

            print("Calculating accuracy on separate test set...")

            test_data = Matrix(arff=eval_parameter)
            if normalize:
                test_data.normalize()

            print("Test set name: {}".format(eval_parameter))
            print("Number of test instances: {}".format(test_data.rows))
            features = Matrix(data, 0, 0, data.rows, data.cols-1)
            labels = Matrix(data, 0, data.cols-1, data.rows, 1)

            start_time = time.time()
            learner.train(features, labels)
            elapsed_time = time.time() - start_time
            print("Time to train (in seconds): {}".format(elapsed_time))

            train_accuracy = learner.measure_accuracy(features, labels)
            print("Training set accuracy: {}".format(train_accuracy))

            test_features = Matrix(test_data, 0, 0, test_data.rows, test_data.cols-1)
            test_labels = Matrix(test_data, 0, test_data.cols-1, test_data.rows, 1)
            confusion = Matrix()
            test_accuracy = learner.measure_accuracy(test_features, test_labels, confusion)
            print("Test set accuracy: {}".format(test_accuracy))

            if print_confusion_matrix:
                print("\nConfusion matrix: (Row=target value, Col=predicted value")
                confusion.print()
                print("")

        elif eval_method == "random":

            print("Calculating accuracy on a random hold-out set...")
            train_percent = float(eval_parameter)
            if train_percent < 0 or train_percent > 1:
                raise Exception("Percentage for random evaluation must be between 0 and 1")
            print("Percentage used for training: {}".format(train_percent))
            print("Percentage used for testing: {}".format(1 - train_percent))

            data.shuffle()

            train_size = int(train_percent * data.rows)
            train_features = Matrix(data, 0, 0, train_size, data.cols-1)
            train_labels = Matrix(data, 0, data.cols-1, train_size, 1)

            test_features = Matrix(data, train_size, 0, data.rows - train_size, data.cols-1)
            test_labels = Matrix(data, train_size, data.cols-1, data.rows - train_size, 1)

            start_time = time.time()
            learner.train(train_features, train_labels)
            elapsed_time = time.time() - start_time
            print("Time to train (in seconds): {}".format(elapsed_time))

            train_accuracy = learner.measure_accuracy(train_features, train_labels)
            print("Training set accuracy: {}".format(train_accuracy))

            confusion = Matrix()
            test_accuracy = learner.measure_accuracy(test_features, test_labels, confusion)
            print("Test set accuracy: {}".format(test_accuracy))

            if print_confusion_matrix:
                print("\nConfusion matrix: (Row=target value, Col=predicted value")
                confusion.print()
                print("")

        elif eval_method == "cross":

            print("Calculating accuracy using cross-validation...")

            folds = int(eval_parameter)
            if folds <= 0:
                raise Exception("Number of folds must be greater than 0")
            print("Number of folds: {}".format(folds))
            reps = 1
            sum_accuracy = 0.0
            elapsed_time = 0.0
            for j in range(reps):
                data.shuffle()
                for i in range(folds):
                    begin = int(i * data.rows / folds)
                    end = int((i + 1) * data.rows / folds)

                    train_features = Matrix(data, 0, 0, begin, data.cols-1)
                    train_labels = Matrix(data, 0, data.cols-1, begin, 1)

                    test_features = Matrix(data, begin, 0, end - begin, data.cols-1)
                    test_labels = Matrix(data, begin, data.cols-1, end - begin, 1)

                    train_features.add(data, end, 0, data.rows - end)
                    train_labels.add(data, end, data.cols-1, data.rows - end)

                    start_time = time.time()
                    learner.train(train_features, train_labels)
                    elapsed_time += time.time() - start_time

                    accuracy = learner.measure_accuracy(test_features, test_labels)
                    sum_accuracy += accuracy
                    print("Rep={}, Fold={}, Accuracy={}".format(j, i, accuracy))

            elapsed_time /= (reps * folds)
            print("Average time to train (in seconds): {}".format(elapsed_time))
            print("Mean accuracy={}".format(sum_accuracy / (reps * folds)))

        else:
            raise Exception("Unrecognized evaluation method '{}'".format(eval_method))

    def parser(self):
        parser = argparse.ArgumentParser(description='Machine Learning System Manager')

        parser.add_argument('-V', '--verbose', action='store_true', help='Print the confusion matrix and learner accuracy on individual class values')
        parser.add_argument('-N', '--normalize', action='store_true', help='Use normalized data')
        parser.add_argument('-L', required=True, choices=['baseline', 'perceptron', 'neuralnet', 'decisiontree', 'knn'], help='Learning Algorithm')
        parser.add_argument('-A', '--arff', metavar='filename', required=True, help='ARFF file')
        parser.add_argument('-E', metavar=('METHOD', 'args'), required=True, nargs='+', help="Evaluation method (training | static <test_ARFF_file> | random <%%_for_training> | cross <num_folds>)")

        return parser


if __name__ == '__main__':
    MLSystemManager().main()

