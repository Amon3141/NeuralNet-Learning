# represents a two layer neural network using back-propagation technique

from functions import *
from layers import *

import numpy as np
from collections import OrderedDict

# representes a two layer neural network using the backward propagation method
class TwoLayerNet_bp:
  # constructs a two layer net
  def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
    # initializes parameters
    self.params = {}
    self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
    self.params["b1"] = np.zeros(hidden_size)
    self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
    self.params["b2"] = np.zeros(output_size)
    
    # creates layers
    self.layers = OrderedDict()
    self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
    self.layers["Relu1"] = ReLU()
    self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])

    self.lastLayer = SoftmaxWithLoss()

  # calculates the output (not the loss)
  def predict(self, x):
    for layer in self.layers.values():
      x = layer.forward(x)
    return x
  
  # calculates the loss
  def loss(self, x, t):
    y = self.predict(x)
    return self.lastLayer.forward(y, t)
  
  # calculates the accuracy
  def accuracy(self, x, t):
    y = self.predict(x)
    y = np.argmax(y, axis=1)
    if t.ndim != 1 : t = np.argmax(t, axis=1)
    accuracy = np.sum(y == t) / float(x.shape[0])
    return accuracy
  
  # calculates the gradient by backward propagation
  def gradient(self, x, t):
    # forward propagation
    # (we don't need the return value, just want to call forward()
    # methods for all the layers)
    self.loss(x, t)

    # backward propagation
    dout = 1
    dout = self.lastLayer.backward(dout)
    layers = list(self.layers.values())
    layers.reverse()
    for layer in layers:
      dout = layer.backward(dout)

    grads = {}
    grads["W1"] = self.layers["Affine1"].dW
    grads["b1"] = self.layers["Affine1"].db
    grads["W2"] = self.layers["Affine2"].dW
    grads["b2"] = self.layers["Affine2"].db

    return grads
  
  def update_params(self, grad, learning_rate):
    for key in self.params:
      assert self.params[key].shape == grad[key].shape, "parameters in different shapes"
      self.params[key] -= learning_rate * grad[key]
