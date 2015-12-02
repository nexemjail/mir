import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import scipy as scipy
import wave


def sigmoid(x,deriv = False):
    if (deriv):
        return x * (1-x)
    return 1 / (1 + np.exp(-x))


X = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]
])

num_of_layers = 3
y = np.array([[0,1,1,0]]).T


syn0 = 2 * np.random.random((3,4)) - 1
syn1 = 2 * np.random.random((4,1)) - 1
syn = [syn0, syn1]
l_delta = [0] * num_of_layers
l_error = [0] * num_of_layers
l = [0] * num_of_layers
l[0] = X
output_layer_index = len(syn)
for j in xrange(5000):
    for i in xrange(1, output_layer_index+1):
        l[i] = sigmoid(np.dot(l[i-1], syn[i-1]))

    l_error[output_layer_index] = y - l[output_layer_index]
    if (j % 1000) == 0:
        print "Error: ", str(np.mean(np.abs(l_error[output_layer_index])))

    l_delta[output_layer_index] = l_error[output_layer_index]\
                                    * sigmoid(l[output_layer_index],True)
    for i in xrange(len(syn)-1,0, -1):
        l_error[i] = l_delta[i+1].dot(syn[i].T)
        l_delta[i] = l_error[i] * sigmoid(l[i],True)

    for i in xrange(len(syn)):
        syn[i] += l[i].T.dot(l_delta[i+1])
print l[2]


