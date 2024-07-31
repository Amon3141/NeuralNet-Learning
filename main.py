# main executing file

# The whole project refers to the codes introduced in "Deep Learning from the Basics" By Koki Saitoh and Shigeo Yushita

# Japanese book URL
# https://www.oreilly.co.jp/books/9784873117584/

# English book URL
# https://www.packtpub.com/en-us/product/deep-learning-from-the-basics-9781800206137

import sys
sys.path.append("/Users/amon/Library/Mobile Documents/com~apple~CloudDocs/download:2023-11-16/deep-learning-from-scratch-master")

import math
import numpy as np
from dataset.mnist import load_mnist
from TwoLayerNet_numgradient import TwoLayerNet_ng
from TwoLayerNet_bpropagation import TwoLayerNet_bp
from SimpleConvNet import SimpleConvNet
#import matplotlib.pyplot as plt

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False, normalize=True, one_hot_label=True)

train_loss_list = []
train_accuracy_list = []

# hyper parameters
iters_num = 1000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

# the input image is 28 x 28 = 784 pixels
#network = TwoLayerNet_bp(input_size=784, hidden_size=50, output_size=10)
network = SimpleConvNet()

"""
# for training
for i in range(iters_num):
  # obtains the minibatch (=100)
  batch_mask = np.random.choice(train_size, batch_size)
  x_batch = x_train[batch_mask]
  t_batch = t_train[batch_mask]

  # calculates the gradients
  grad = network.gradient(x_batch, t_batch)

  # updates the parameters
  network.update_params(grad, learning_rate)

  # record the learning process
  loss = network.loss(x_batch, t_batch)
  accuracy = network.accuracy(x_batch, t_batch)
  train_loss_list.append(loss)
  train_accuracy_list.append(accuracy)

  if ((i+1) % 10 == 0):
    print(f"{i+1} iterations done")

network.save_params("params.pkl")
print("Saved Network Parameters!")
"""

network.load_params("params.pkl")
print("Loaded Network Parameters!")

test_size = t_test.shape[0]
batch_iters_num = test_size // batch_size

accuracy = 0.0
for i in range(batch_iters_num):
  batch_mask = np.arange(100*i, 100*(i+1))
  x_batch  = x_test[batch_mask]
  t_batch = t_test[batch_mask]
  accuracy += network.accuracy(x_batch, t_batch)

if test_size % batch_iters_num != 0:
  last_x_batch = x_test[100*batch_iters_num:]
  last_t_batch = t_test[100*batch_iters_num:]
  accuracy += network.accuracy(last_x_batch, last_t_batch)
  batch_iters_num += 1

accuracy /= (batch_iters_num)

# print the train & test accuracies
#print("train_accuracy: {}".format(accuracy))
print("test_accuracy: {}".format(accuracy))

"""
# plot the learning process
x = np.arange(iters_num)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.plot(x, train_loss_list)
ax1.set_title("Loss")
ax1.set_xlabel("Iterations")
ax1.set_ylabel("Loss")

ax2.plot(x, train_accuracy_list)
ax2.set_title("Accuracy")
ax2.set_xlabel("Iterations")
ax2.set_ylabel("Accuracy")

plt.tight_layout()
plt.show()
"""
