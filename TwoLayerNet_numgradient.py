# represents a two layer neural network using numerical gradient

from functions import *
from gradient import *

import numpy as np

# represents a neural network with one hidden layer (overall two layers)
class TwoLayerNet_ng:
  # constructs a two layer net with the number of input nodes, nodes in the hidden
  # layer, and the output nodes AND
  # initialize the parameters (weights and biases)
  def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
    self.params = {}
    self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)
  
  # predicts the output (y) from the input (x)
  # INP: x : numpy ndarray of the size (batch_size, input_size)
  # RET: y : numpy ndarray of the size (batch_size, output_size)
  def predict(self, x):
    W1, W2 = self.params['W1'], self.params['W2']
    b1, b2 = self.params['b1'], self.params['b2']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    y = softmax(a2)

    return y
  
  # calculates the loss using the cross entropy error
  # INP: x in the input, t is the teaching data
  def loss(self, x, t):
    y = self.predict(x)
    return cross_entropy_error(y, t)
  
  # calculates the accuracy (classification)
  # INP: x in the input, t is the teaching data
  def accuracy(self, x, t):
    y = self.predict(x)
    y = np.argmax(y, axis=1)
    t = np.argmax(t, axis=1)

    accuracy = np.sum(y == t) / float(x.shape[0])
    return accuracy
  
  def numerical_gradient(self, x, t):
    loss_W = lambda W: self.loss(x, t) 
    # W is a fake parameter (the purpose of this function is just to call self.loss(x, t))

    grads = {}
    grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
    grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
    grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
    grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

    return grads
  
  def update_params(self, grad, learning_rate):
    for key in self.params:
      assert self.params[key].shape == grad[key].shape, "parameters in different shapes"
      self.params[key] -= learning_rate * grad[key]
