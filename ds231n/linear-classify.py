import numpy as np
xrange = range

import numpy
import os
import sys


def L_i(x, y, W):
    """
    unvectorized version. Compute the multiclass svm loss for a single example (x, y)
    - x is a column vector representing an image (e.g. 3073 x 1 in CIFAR-10)
      with an appended bias dimension in the 3073-rd position (i.e. bias trick)
    - y is an integer giving index of correct class (e.g. between 0 and 9 in CIFAR-10)
    - W is the weight matrix (e.g. 10 x 3073 in CIFAR-10)
    """
    delta = 1.0  # see notes about delta later in this section
    # scores becomes of size 10 x 1, the scores for each class
    scores = W.dot(x)
    correct_class_score = scores[y]
    D = W.shape[0]  # number of classes, e.g. 10
    loss_i = 0.0
    for j in xrange(D):  # iterate over all wrong classes
        if j == y:
            # skip for the true class to only loop over incorrect classes
            continue
        # accumulate loss for the i-th example
        loss_i += max(0, scores[j] - correct_class_score + delta)
    return loss_i


def L_i_vectorized(x, y, W):
    """
    A faster half-vectorized implementation. half-vectorized
    refers to the fact that for a single example the implementation contains
    no for loops, but there is still one loop over the examples (outside this function)
    """
    delta = 1.0
    scores = W.dot(x)
    # compute the margins for all classes in one vector operation
    margins = np.maximum(0, scores - scores[y] + delta)
    # on y-th position scores[y] - scores[y] canceled and gave delta. We want
    # to ignore the y-th position and only consider margin on max wrong class
    margins[y] = 0
    loss_i = np.sum(margins)
    return loss_i


def L(X, y, W):
    """
    full-vectorized implementation:
    - X holds all the training examples as columns (e.g. 3073 x 50,000 in CIFAR-10)
    - y is array of integers specifying correct class (e.g. 50,000-D array)
    - W are weights (e.g. 10 x 3073)
    """
    # evaluate loss over all examples in X without using any for loops
    delta = 1.0
    scores = W.dot(X)
    #print('\nscores:\n', scores)

    #print("\n", scores[y, :])
    #print("\n", scores - scores[y, :])

    margins = np.maximum(0, scores - scores[y, :] + delta)

    #print('\nmargins before:\n', margins)
    margins[y, :] = 0
    #print('\nmargins after:\n', margins)

    loss = np.sum(margins, axis=0)
    return loss


if __name__ == '__main__':
    x = np.ones(4)
    y = 0
    W = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [-1, -1, -1, -1]])
    print('x.shape, W.shape:', x.shape, W.shape)

    print(L_i(x, y, W))
    print(L_i_vectorized(x, y, W))

    X = np.ones((4, 3))
    X[:, 1] += 1
    X[:, 2] += 2
    print(X)
    print('\nloss:', L(X, y, W))
