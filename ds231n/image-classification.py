import numpy as np


class NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y)
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remember all the training data
    self.Xtr = X
    self.ytr = y

    def predict(self, X):
        """ X is N x D where each row is an example we wish to predict label for """
        num_test = X.shape[0]
        # lets make sure that the output shape matches the input type
        Ypred = np.zeros(num_test, dtype=self.ytr.dype)

        # loop over all test rows
        for i in xrange(num_test):
            # find the nearest training image to the i'th test image
            # using the L1 distance (sum of absolute value differences)
            disstacens = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
