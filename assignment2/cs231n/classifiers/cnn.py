import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    
    # Unpack the dimensions of the input data
    C, H, W = input_dim

    #mp_out_height = H / 2 # Max pool 2,2 means conv settings are changed
    #mp_out_width = W / 2 # to make the height and width of the output halve !
    #
    #conv_out_dims = (num_filters, conv_out_height, conv_out_width)
    mp_out_dims = (num_filters, H/2, W/2)
                                            
    conv_W_dims = (num_filters, C, filter_size, filter_size)
    hidden_W_size = num_filters * (H/2) * (W/2)
    
    #print 'conv_W_dims = {}'.format(conv_W_dims)
    #print 'conv weight shape = {}'.format(conv_W_dims)
    #print 'conv out shape = {}'.format(conv_out_dims)
    #print 'maxpool out shape = {}'.format(mp_out_dims)
    #print 'hidden_dim = {}'.format(hidden_dim)
    
    #print conv_W_dims
    #print (mp_out_dims, hidden_dim)
    
    #print mp_out_dims
    #
    self.params['W1'] = weight_scale * np.random.randn(*conv_W_dims)
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = weight_scale * np.random.randn(hidden_W_size, hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b3'] = np.zeros(num_classes)
    #
    #
    #print 'W1 shape = {}'.format(self.params['W1'].shape)
    #print 'b1 shape = {}'.format(self.params['b1'].shape)
    #print 'W2 shape = {}'.format(self.params['W2'].shape)
    #print 'b2 shape = {}'.format(self.params['b2'].shape)
    #print 'W3 shape = {}'.format(self.params['W3'].shape)
    #print 'b3 shape = {}'.format(self.params['b3'].shape)
    #
    #print ' '
    #print 'W1 = {}'.format(self.params['W1'])
    #print 'b1 = {}'.format(self.params['b1'])
    #print 'W2 = {}'.format(self.params['W2'])
    #print 'b2 = {}'.format(self.params['b2'])
    #print 'W3 = {}'.format(self.params['W3'])
    #print 'b3 = {}'.format(self.params['b3'])

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    
    
    crp_out, crp_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    #print 'crp_out.shape = {}'.format(crp_out.shape)
    
    ar_out, ar_cache = affine_relu_forward(crp_out, W2, b2)
    #print ar_out.shape
    
    a_out, a_cache = affine_forward(ar_out, W3, b3)
    #print a_out.shape
    
    scores = a_out
    
    # Calculate loss (both data and regularization loss)
    data_loss, dx = softmax_loss(scores, y)
    reg_loss = 0
    reg_loss += 0.5 * self.reg * np.sum(W1*W1) 
    reg_loss += 0.5 * self.reg * np.sum(W2*W2) 
    reg_loss += 0.5 * self.reg * np.sum(W3*W3) 
    
    loss = data_loss + reg_loss
    
    #print 'loss = {}'.format(loss)
    #print crp_out[0]
    #print ar_out[0]
    #print a_out[0]

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    grads = {} # Loss is already calculated above
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    
    
    a_dx, a_dw, a_db = affine_backward(dx, a_cache)
    
    ar_dx, ar_dw, ar_db = affine_relu_backward(a_dx, ar_cache)
    
    crp_dx, crp_dw, crp_db =  conv_relu_pool_backward(ar_dx, crp_cache)

    grads['W1'] = crp_dw
    grads['b1'] = crp_db
    grads['W2'] = ar_dw
    grads['b2'] = ar_db
    grads['W3'] = a_dw
    grads['b3'] = a_db
    
    # Add regularization for weights
    grads['W1'] += self.reg * grads['W1']
    grads['W2'] += self.reg * grads['W2']
    grads['W3'] += self.reg * grads['W3']

    #print grads
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
