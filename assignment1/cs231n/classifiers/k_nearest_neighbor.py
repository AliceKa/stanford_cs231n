import numpy as np

class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      for j in xrange(num_train):
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
        #print self.X_train[j:,]
        #print X
        #print 'Top of loop, i = {}, j = {}'.format(i, j)
        #print self.X_train[j,:]
        #print X[i,:]
        differences = self.X_train[j,:] - X[i, :]
        #print differences
        squareDifferences = differences ** 2
        #print squareDifferences
        sumSquareDifferences = np.sum(squareDifferences, axis=-1)
        #print sumSquareDifferences
        euclideanDistance = np.sqrt(sumSquareDifferences)
        #print euclideanDistance
        #print 'Bottom of loop'
        dists[i, j] = euclideanDistance
        
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      #if (i%100 == 0):
      #    print 'Calculating i = {}'.format(i)
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
      testVector = X[i,:]
      testVector = testVector[np.newaxis,:]
      #print testVector.shape
      # Now take the dot product of the testing vector with the training matrix
      #print self.X_train.shape
      #print testVector.shape
      testDifference = self.X_train - testVector
      testDifference = testDifference ** 2
      testDifference = testDifference.sum(axis=-1)
      testDifference = np.sqrt(testDifference)
      #print testDifference.shape
      #testDotProduct = testDotProduct[np.newaxis,:]
      #print testDotProduct.shape
      dists[i,:] = testDifference
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################

    # This has way too high memory consumption !    
    #print 'X_train shape is {}, X shape is {}'.format(self.X_train.shape, X[:,:,np.newaxis].shape)
    #dists = self.X_train.T - X[:,:,np.newaxis]
    #print dists.shape
    #distsSquared = dists ** 2
    #print distsSquared.shape
    #distsSum = distsSquared.sum(axis=1)
    #print distsSum.shape
    #distsSqrt = np.sqrt(distsSum)
    #dists = distsSqrt
    #print ' '
    #print X.shape
    #print self.X_train.shape
    
    xTrainNorm = np.diag(np.dot(self.X_train, self.X_train.T))
    xNorm = np.diag(np.dot(X, X.T))
    xDiffNorm = np.dot(X, self.X_train.T)
    
    xTrainNorm = xTrainNorm[np.newaxis,:]
    xNorm = xNorm[:,np.newaxis]
    
    #print ' '
    #print xTrainNorm.shape
    #print xNorm.shape
    #print xDiffNorm.shape
    
    dists = np.sqrt(xTrainNorm - (xDiffNorm * 2) + xNorm)
    #print dists.shape
    
    
    # The calculation below uses way too much memory !
    #dists = np.sqrt(((self.X_train.T - X[:,:,np.newaxis]) ** 2).sum(axis=1))
    
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in xrange(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
      testArgsorted = dists[i].argsort()
      #print testArgsorted
      closest_y = self.y_train[testArgsorted]
      #print closest_y
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      matching_labels = dict()
      final_class = -1
      
      for y in closest_y:
          if y in matching_labels:
              matching_labels[y] += 1
          else :
              matching_labels[y] = 1
              
          #print 'Example {} Found nearest {} examples, class is {}, dict is {}'.format(i, k, y, matching_labels)
          if (matching_labels[y] == k):
              final_class = y
              break    
      
      y_pred[i] = y
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################

    return y_pred

