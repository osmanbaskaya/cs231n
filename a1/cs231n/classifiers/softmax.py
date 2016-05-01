import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    N = X.shape[0]
    C = W.shape[1]

    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    for i in xrange(N):
        score = np.dot(X[i], W)
        max_score = np.max(score)
        score -= max_score
        score = np.exp(score)
        normalized_score = score / np.sum(score)
        correct_class_score = normalized_score[y[i]]
        loss += -np.log(correct_class_score)
        for j in xrange(C):
            dW[:, j] += (normalized_score[j] - (y[i] == j)) * X[i].T

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    loss /= N
    loss += 0.5 * reg * np.sum(W * W)

    dW /= N
    dW += reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################

    N = X.shape[0]
    scores = np.dot(X, W).T
    scores -= np.max(scores, axis=0)  # for computational stability.
    scores = np.exp(scores)
    total_scores = np.sum(scores, axis=0)
    normalized_scores = scores / total_scores
    loss += -np.log(normalized_scores[y, np.arange(0, N)]).sum()
    loss += 0.5 * reg * np.sum(W * W)
    loss /= N

    normalized_scores[y, np.arange(0, N)] -= 1
    dW += np.dot(normalized_scores, X).T / N
    dW += reg * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
