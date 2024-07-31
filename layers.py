# collection of classes for layers that build the neural net

import numpy as np
from functions import *

# Simple layers (not used in the actual network)
# AdditionLayer
class AdditionLayer:
  def __init__(self):
    pass

  def forward(self, x, y):
    out = x + y
    return out
  
  def backward(self, dout):
    dx = dout * 1
    dy = dout * 1
    return (dx, dy)
  
# MultiplicationLayer
class MultiplicationLayer:
  def __init__(self):
    self.x = None
    self.y = None

  def forward(self, x, y):
    self.x = x
    self.y = y
    out = self.x * self.y
    return out
  
  def backward(self, dout):
    dx = dout * self.y
    dy = dout * self.x
    return (dx, dy)
  
# Layers used in the neural network
# Activation function layers
# ReLU layer
class ReLU:
  def __init__(self):
    self.mask = None # mask for elements <= 0
  
  def forward(self, x):
    self.mask = (x <= 0)
    out = x.copy()
    out[self.mask] = 0
    return out
  
  def backward(self, dout):
    dout[self.mask] = 0
    dx = dout
    return dx
  
# Sigmoid layer
class Sigmoid:
  def __init__(self):
    self.out = None
  
  def forward(self, x):
    out = 1 / (1 + np.exp(-x))
    self.out = out
    return out
  
  def backward(self, dout):
    dx = dout * self.out * (1.0 - self.out)
    return dx
    
# Output function (with loss function) layers
# identity with mean square error layer
class IdentityWithLoss:
  def __init__(self):
    self.y = None
    self.t = None
  
  def forward(self, x, t):
    self.y = x
    self.t = t
    out = square_error(self.y, self.t)
    return out

  def backward(self, dout=1):
    batch_size = self.t.shape[0]
    dx = dout * (self.y - self.y) / batch_size
    return dx
  
# Softmax with cross entropy error layer
class SoftmaxWithLoss:
  def __init__(self):
    self.y = None
    self.t = None
  
  def forward(self, x, t):
    self.y = softmax(x)
    self.t = t
    out = cross_entropy_error(self.y, self.t)
    return out
  
  def backward(self, dout=1):
    batch_size = self.t.shape[0]
    dx = dout * (self.y - self.t) / batch_size
    return dx
  
# Affine layer
class Affine:
  def __init__(self, W, b):
    self.W = W
    self.b = b
    self.x = None
    self.dW = None
    self.db = None
  
  def forward(self, x):
    self.x = x
    try:
      out = np.dot(x, self.W) + self.b
    except ValueError as e: #!!!
      x = x.reshape(-1, self.W.shape[0])
      out = np.dot(x, self.W) + self.b
    
    return out
  
  def backward(self, dout):
    dx = np.dot(dout, self.W.T)
    self.dW = np.dot(self.x.T, dout)
    self.db = np.sum(dout, axis=0)
    return dx
  
# CNN-related layers
# Convolution layer
class Convolution:
  def __init__(self, W, b, stride=1, pad=0): # W is the filter
    self.W = W
    self.b = b
    self.stride = stride
    self.pad = pad

    self.x = None
    self.col = None
    self.col_W = None # column(2D) version of the filter(self.W)

    self.dW = None
    self.db = None
  
  def forward(self, x):
    FN, C, FH, FW = self.W.shape
    N, C, H, W = x.shape
    out_h = 1 + (H + 2*self.pad - FH) // self.stride
    out_w = 1 + (W + 2*self.pad - FW) // self.stride

    col = im2col(x, FH, FW, self.stride, self.pad) # col.shape = (N*OH*OW, C*FH*FW)
    col_W = self.W.reshape(FN, -1).T # col_W.shape = (C*FH*FW, FN)
    # OH*OW -> how many times a filter is applied to one image data? (N*which is the total number of times  
    #  the filter (kernel) is applied to x : the input data to the Convolution layer)
    # C*FH*FW -> how many cells are included in one filter? (the volume of the filter recutangular)
    out = np.dot(col, col_W) + self.b

    out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2) 
    # out: (N, -1(FN), out_h, out_w)

    self.x = x
    self.col = col
    self.col_W = col_W

    return out
  
  def backward(self, dout):
    FN, C, FH, FW = self.W.shape
    dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)
    # dout : (N, FN, out_h, out_w), dout.transpose(0, 2, 3, 1) : (N, out_h, out_w, FN), 
    # dout.transpose(0, 2, 3, 1).reshape(-1, FN) : (N, out_h, out_w, FN)

    self.db = np.sum(dout, axis=0)
    self.dW = np.dot(self.col.T, dout)
    self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

    dcol = np.dot(dout, self.col_W.T)
    dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

    return dx

# Pooling layer
class Pooling:
  def __init__(self, pool_h, pool_w, stride=2, pad=0):
    self.pool_h = pool_h
    self.pool_w = pool_w
    self.stride = stride
    self.pad = pad

    self.out_h = None
    self.out_w = None

    self.argmax = None
    self.x = None
  
  def forward(self, x):
    N, C, H, W = x.shape
    out_h = 1 + (H + 2*self.pad - self.pool_h) // self.stride
    out_w = 1 + (W + 2*self.pad - self.pool_w) // self.stride

    col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
    col = col.reshape(-1, self.pool_h*self.pool_w)

    argmax = np.argmax(col, axis=1)
    out = np.max(col, axis=1)
    out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

    self.argmax = argmax
    self.x = x
    self.out_h = out_h
    self.out_w = out_w

    return out
  
  def backward(self, dout):
    try:
      dout = dout.transpose(0, 2, 3, 1)
    except ValueError as e: #!!!
      dout = dout.reshape(dout.shape[0], -1, self.out_h, self.out_w).transpose(0, 2, 3, 1)

    pool_size = self.pool_h * self.pool_w
    dmax = np.zeros((dout.size, pool_size))
    dmax[np.arange(self.argmax.size), self.argmax.flatten()] = dout.flatten()
    dmax = dmax.reshape(dout.shape + (pool_size,))

    dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
    dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

    return dx