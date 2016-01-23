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

  num_train = X.shape[0]
  num_dim = X.shape[1]
  num_class = W.shape[1]
  scores = np.zeros((num_train, num_class))
  dScores = np.zeros((num_train, num_class))

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  #print W[:3,:]
  #print X[:3,:]
  #print y[:3]
  
  biggestVal = -1
  
  # Matrix-multiply the X and W matrices, remember largest value
  for i in xrange(num_train):
    for j in xrange(num_class):
      dotSum = 0
      for k in range(num_dim):
          dotSum += X[i, k] * W[k, j]
      
      scores[i, j] = dotSum
      if dotSum > biggestVal:
        biggestVal = dotSum
      
  # Subtract maximum value to keep all scores < 0 to prevent the exponential
  # step blowing up
  scores -= biggestVal
  
  # Iterate over each example, take the exponent of the scores, then
  # normalise across example tso
  for i in xrange(num_train):
    row_sum = 0
    
    # Calculate score for given example and class
    for j in xrange(num_class):
      scores[i, j] = np.exp(scores[i, j])
      row_sum += scores[i, j]
    
    # Divide by the sum of the row entries
    for j in xrange(num_class):
      scores[i, j] /= row_sum
    
    # Calculate the loss using -log10(P(correct example Probability))
  loss = 0
  for idx, row in enumerate(scores):
    correctClass = y[idx]
    delta_loss = -1 * np.log(row[correctClass])
    loss += delta_loss
  
  # Normalise loss for number of examples
  loss /= num_train

  # First step of gradient calculation, subract 1 from the correct class 
  # probability
  for rowNum in xrange(num_train):
    correctClass = y[rowNum]
    scores[rowNum, correctClass] -= 1

  # Normalise across the training set
  scores /= num_train
  # Multiply X and scores to get change in gradient.
  dW = np.dot(X.T, scores)

  ## Add regularization loss too
  regLoss = 0
  for i in xrange(num_dim):
    for j in xrange(num_class):
      regLoss += W[i,j] * W[i, j]

  regLoss = 0.5 * reg * regLoss
  loss += regLoss
  dW += reg * W
  
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
  num_examples = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  # evaluate class scores, [N x K]
  scores = np.dot(X, W)
  scores -= np.max(scores)
  
  # compute the class probabilities
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

  # compute the loss: average cross-entropy loss and regularization
  corect_logprobs = -np.log(probs[range(num_examples),y])
  data_loss = np.sum(corect_logprobs)/num_examples
  reg_loss = 0.5*reg*np.sum(W*W)
  loss = data_loss + reg_loss
  
  # compute the gradient on scores
  dscores = probs
  dscores[range(num_examples),y] -= 1
  dscores /= num_examples

  # Update gradient
  dW = np.dot(X.T, dscores)
  dW += reg*W # regularization gradient
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

