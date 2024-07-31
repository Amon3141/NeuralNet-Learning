# collection of functions relating the calculation of gradient

import numpy as np

# gradient is a vector at a certain point
# which is (df/dx_1, df/dx_2, ..., df/dx_n)

# f is a function that takes a np.ndarray x. f: ndarray -> int
# x is a point to calculate the gradient: (x_1, x_2, ..., x_n) 
def numerical_gradient(f, x):
  gradient = np.zeros_like(x)
  h = 1e-4

  with np.nditer(x, flags=["multi_index"], op_flags=["readwrite"]) as it:
    for a in it:
      temp = a
      #calculate f(x+h)
      a[...] = temp + h
      f_xph = f(x)
      #calculate f(x-h)
      a = temp - h
      f_xmh = f(x)
      #partial derivative
      grad_i = (f_xph - f_xmh) / 2*h
      gradient[it.multi_index] = grad_i
      a = temp
  
  return gradient