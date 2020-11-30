from sklearn.model_selection import ShuffleSplit, KFold, StratifiedKFold, StratifiedShuffleSplit
import numpy as np


class KFoldsSplitter(object):
    def __init__(self, X, y, k=5, ratio=0.9, stratified=False, random_state=0):
        """
        A class that is able to split train_dev and test using the 'ratio', and do K-fold on the train_dev
        :param X: X
        :param y: y
        :param k: nun of fold
        :param ratio: train_dev to test
        :param stratified: option to keep stratified w/r to the label distribution
        """
        self.X_train_dev = None
        self.y_train_dev = None
        self.X_test = None
        self.y_test = None
        self.test_size = 1 - ratio
        self.k = k
        self.random_state = random_state
        self.result = []
        self.is_stratified = stratified
        self.splitter = None
        self.split(X, y)

    def split(self, X, y):
        if self.is_stratified:
            ss = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)
            train_index, test_index = next(ss.split(X, y))
            X_train_dev, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
            y_train_dev, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

            self.X_train_dev = X_train_dev
            self.y_train_dev = y_train_dev
            self.X_test = X_test
            self.y_test = y_test
            skf = StratifiedKFold(n_splits=self.k, random_state=self.random_state)
            self.splitter = skf.split(self.X_train_dev, self.y_train_dev)

        else:
            ss = ShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)
            train_index, test_index = next(ss.split(y))
            X_train_dev, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
            y_train_dev, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

            self.X_train_dev = X_train_dev
            self.y_train_dev = y_train_dev
            self.X_test = X_test
            self.y_test = y_test
            kf = KFold(n_splits=self.k, random_state=self.random_state)
            self.splitter = kf.split(self.y_train_dev)

    def next_fold(self):
        train_index, dev_index = next(self.splitter)
        X_train, X_dev = [self.X_train_dev[i] for i in train_index], [self.X_train_dev[i] for i in dev_index]
        y_train, y_dev = [self.y_train_dev[i] for i in train_index], [self.y_train_dev[i] for i in dev_index]
        return X_train, y_train, X_dev, y_dev

    def get_test(self):
        return self.X_test, self.y_test

    def get_train_dev(self):
        return self.X_train_dev, self.y_train_dev

    def add_result(self, y_test_pred):
        self.result.append(y_test_pred)

    def get_voting_result(self):
        def find_majority(k):
            myMap = {}
            maximum = ('', 0)  # (occurring element, occurrences)
            for n in k:
                if n in myMap:
                    myMap[n] += 1
                else:
                    myMap[n] = 1
                # Keep track of maximum on the go
                if myMap[n] > maximum[1]: maximum = (n, myMap[n])
            return maximum

        all_preds = np.stack(self.result, axis=0)

        if len(all_preds.shape) == 2:
            shape = all_preds[0].shape
            mj = np.zeros(shape)
            for m in range(shape[0]):
                    mj[m] = find_majority(np.asarray(all_preds[:, m]).reshape((-1)))[0]
            return mj
        elif len(all_preds.shape) == 3:
            shape = all_preds[0].shape
            mj = np.zeros(shape)
            for m in range(shape[0]):
                for n in range(shape[1]):
                    mj[m, n] = find_majority(np.asarray(all_preds[:, m, n]).reshape((-1)))[0]
        else:
            raise Exception('The shape of the predictions:', all_preds.shape, 'can not be handled')

    def get_avg_result(self):
        all_preds = np.stack(self.result, axis=0)
        return np.mean(all_preds, axis=0)

    def get_metric(self, mode='avg', metric_fn=None):
        if mode == 'avg':
            return metric_fn(self.y_test, self.get_avg_result())
        elif mode == 'mj':
            return metric_fn(self.y_test, self.get_voting_result())
        else:
            raise Exception("Mode is not correctly define, must be one 'avg' or 'mj'!")


class KFoldsSplitterNoTest(KFoldsSplitter):
    def __init__(self, X, y, k=5, ratio=0.9, stratified=False, random_state=0):
        super(KFoldsSplitterNoTest, self).__init__(X, y, k, ratio, stratified, random_state)

        """
        A class that is able to split train_dev and test using the 'ratio', and do K-fold on the train_dev
        :param X: X
        :param y: y
        :param k: nun of fold
        :param ratio: train_dev to test
        :param stratified: option to keep stratified w/r to the label distribution
        """

    def split(self, X, y):
        if self.is_stratified:
            self.X_train_dev = X
            self.y_train_dev = y
            skf = StratifiedKFold(n_splits=self.k, random_state=self.random_state)
            self.splitter = skf.split(self.X_train_dev, self.y_train_dev)

        else:
            ss = ShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)
            train_index, test_index = next(ss.split(y))
            X_train_dev, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
            y_train_dev, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

            self.X_train_dev = X_train_dev
            self.y_train_dev = y_train_dev
            self.X_test = X_test
            self.y_test = y_test
            kf = KFold(n_splits=self.k, random_state=self.random_state)
            self.splitter = kf.split(self.y_train_dev)

    def get_test(self):
        raise NotImplementedError
