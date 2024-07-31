# represents a simple convolutional network class

from functions import *
from layers import *

import pickle
import numpy as np
from collections import OrderedDict

# edited some parts of the code to fit to this particular example.
# adjusted the input/output shapes of ndarrays at "!!!"" in SimpleConvNet.py and layers.py

# represents a network of conv - relu - pool - affine - relu - affine - softmax(with loss)
class SimpleConvNet:
  def __init__(self, input_dim=(1, 28, 28), conv_param={"filter_num":30, "filter_size":5, "pad":0, "stride":1}, hidden_size=100, output_size=10, weight_init_std=0.01):
    filter_num = conv_param["filter_num"]
    filter_size = conv_param["filter_size"]
    filter_pad = conv_param["pad"]
    filter_stride = conv_param["stride"]
    input_size = input_dim[1]
    conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
    # suppose that pool_h = pool_h = 2, stride = 2↓
    pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2)) 
    
    self.params = {}
    self.params["W1"] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
    self.params["b1"] = np.zeros(filter_num)
    self.params["W2"] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
    self.params["b2"] = np.zeros(hidden_size)
    self.params["W3"] = weight_init_std * np.random.randn(hidden_size, output_size)
    self.params["b3"] = np.zeros(output_size)

    self.layers = OrderedDict()
    self.layers["Conv1"] = Convolution(self.params["W1"], self.params["b1"], filter_stride, filter_pad)
    self.layers["Relu1"] = ReLU()
    self.layers["Pool1"] = Pooling(pool_h=2, pool_w=2, stride=2, pad=0)
    self.layers["Affine1"] = Affine(self.params["W2"], self.params["b2"])
    self.layers["Relu2"] = ReLU()
    self.layers["Affine2"] = Affine(self.params["W3"], self.params["b3"])
    
    self.last_layer = SoftmaxWithLoss()
  
  def predict(self, x):
    for layer in self.layers.values():
      x = layer.forward(x)
    return x
  
  def loss(self, x, t):
    y = self.predict(x)
    e = self.last_layer.forward(y, t)
    return e
  
  def gradient(self, x, t):
    self.loss(x, t)

    dout = self.last_layer.backward(dout=1)
    layers = list(self.layers.values())
    layers.reverse()
    for layer in layers:
      dout = layer.backward(dout)
    
    grads = {}
    grads["W1"] = self.layers["Conv1"].dW
    grads["b1"] = self.layers["Conv1"].db
    grads["W2"] = self.layers["Affine1"].dW
    grads["b2"] = self.layers["Affine1"].db
    grads["W3"] = self.layers["Affine2"].dW
    grads["b3"] = self.layers["Affine2"].db

    return grads
  
  def accuracy(self, x, t):
    y = self.predict(x)
    y = np.argmax(y, axis=1)
    if t.ndim != 1 : t = np.argmax(t, axis=1)
    accuracy = np.sum(y == t) / float(x.shape[0])
    return accuracy
  
  def update_params(self, grads, learning_rate):
    for key in self.params:
      if (self.params[key].shape != grads[key].shape): #!!!
        grads[key] = grads[key].reshape(self.params[key].shape) #!!!
      self.params[key] -= learning_rate * grads[key]

  def save_params(self, file_name="params.pkl"):
    params = {}
    for key, val in self.params.items():
        params[key] = val
    with open(file_name, 'wb') as f:
        pickle.dump(params, f)

  def load_params(self, file_name="params.pkl"):
      with open(file_name, 'rb') as f:
          params = pickle.load(f)
      for key, val in params.items():
          self.params[key] = val

      for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
          self.layers[key].W = self.params['W' + str(i+1)]
          self.layers[key].b = self.params['b' + str(i+1)]