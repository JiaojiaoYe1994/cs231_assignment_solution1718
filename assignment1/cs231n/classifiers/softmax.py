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
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  L = np.zeros(num_classes)

  for i in range(num_train):
    scores = X[i].dot(W)
    L = np.exp(scores)
    loss -= np.log( L[y[i]] / np.sum(L))
    for j in range(num_classes):
        # grad for correct class
        if j == y[i]:
            dW[:, j] -= X[i].T * (np.sum(L) - L[j])/np.sum(L)
        # grad for incorrect class
        else:
            dW[:, j] += X[i].T * (L[j] / np.sum(L))
  loss /= num_train
  loss += reg* np.sum(W*W)
  dW /= num_train
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  num_train = X.shape[0]
  scores = X.dot(W)# (N , C)
  L = np.exp(scores)
  loss -= np.sum(np.log( L[range(num_train), y] / np.sum(L, axis=1)))
  loss /= num_train
  loss += reg* np.sum(W*W)
    
  mask = np.zeros_like(scores)
  mask[range(num_train), y] = 1
  dW += X.T.dot(L/ np.repeat(np.sum(L, axis=1)[:, np.newaxis], 10, axis=1 ) - mask )
  dW /= num_train
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

