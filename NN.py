import numpy as np
import pandas


def sigmoid_derivative(x):
    return sigmoid(x) * (1-sigmoid(x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class NeuralNetwork(object):
    def __init__(self, learning_rate = 0.01, num_of_hidden_layers = 8, hidden_layer_size = 50):
        self._num_of_layers = num_of_hidden_layers + 2
        self._num_of_hidden_layers = num_of_hidden_layers
        self._hidden_layer_size = hidden_layer_size
        self._syn_count = self._num_of_layers - 1
        self._alpha = learning_rate
        self._data = []
        self._results = []
        self._training_size = 0
        self._input_layer_size = 0
        self._output_layer_size = 0
        self._data_dir = "synapses"
        self._synapses = None

    def save_synapse_to_file(self, folder_name, synapses):
        for i in xrange(self._syn_count):
            realpath = "{0}/syn{1}.csv".format(folder_name,i)
            with open(realpath, 'w'):
                frame = pandas.DataFrame(synapses[i])
                frame.to_csv(realpath)

    def read_synapses(self,folder_name):
        syn = []
        for i in xrange(self._syn_count):
            realpath = "{0}/syn{1}.csv".format(folder_name,i)
            with open(realpath, 'r'):
                syn.append(pandas.DataFrame.from_csv(realpath))
        return syn

    def fit(self, data_array, result_array):
        self._data = data_array
        self._training_size = len(data_array)
        self._results = result_array
        self._input_layer_size = data_array[0].shape[0]
        self._output_layer_size = result_array[0].shape[0]
        self._learn()

    def _learn(self):
        np.random.seed(0)
        from_input_syn = 2 * np.random.random((self._input_layer_size, self._hidden_layer_size)) - 1
        to_output_syn = 2 * np.random.random((self._hidden_layer_size, self._output_layer_size)) - 1
        syn = [0] * self._syn_count

        syn[0], syn[self._syn_count - 1] = from_input_syn, to_output_syn

        for i in xrange(1,self._syn_count-1):
            syn[i] = 2 * np.random.random((self._hidden_layer_size, self._hidden_layer_size)) - 1

        l = [0] * self._num_of_layers
        l[0] = None
        l_error = [None] * self._num_of_layers
        l_delta = [None] * self._num_of_layers

        for j in xrange(self._training_size):

            l[0] = self._data[j]
            for i in xrange(1,self._num_of_layers):
                try:
                    l[i] = sigmoid(np.dot(l[i-1], syn[i-1]))
                except:
                    i = 5;
                '''
                if do_dropout:
                    k = np.random.binomial(
                                [np.ones((l[i].shape[0],syn[i-1].shape[1]))],\
                            1 - dropout_percent)[0]
                    l[i] *= k * (1.0 / (1 - dropout_percent))
                '''
            l_error[self._num_of_layers - 1] = l[self._num_of_layers - 1] - self._results[j]
            #if j % 5 == 0:
                #print "iteration", str(j+1)
            print "Error: ", str(np.mean(np.abs(l_error[self._num_of_layers-1])))
            l_delta[self._num_of_layers - 1] = l_error[self._num_of_layers-1] * sigmoid_derivative(l[self._num_of_layers-1])

            for i in xrange(self._num_of_layers - 2 ,-1,-1):
                l_error[i] = np.dot(l_delta[i+1],syn[i].T)
                l_delta[i] = l_error[i] * sigmoid_derivative(l[i])

            for i in xrange(self._syn_count):
                syn[i] -= self._alpha * l[i].T.dot(l_delta[i])

        self._synapses = syn
        self.save_synapse_to_file(self._data_dir,syn)
        #print l[num_of_layers - 1]

    def predict(self, data):
        l = [0] * self._num_of_layers
        l[0] = data
        for i in xrange(1,self._num_of_layers):
            l[i] = sigmoid(np.dot(l[i-1], self._synapses[i-1]))
        return l[self._num_of_layers - 1]



'''
X = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1],
])
y = np.array([[0, 1, 1, 0]]).T
num_of_layers = 3
num_of_hidden_layers = num_of_layers - 2
syn_count = num_of_layers - 1
hidden_layer_size = 15
alphas = [1, 2, 3, 4, 5]
dropout_percent = 0.5
do_dropout = False
for alpha in alphas:
    print "Alpha is ",alpha
    np.random.seed(0)
    from_input_syn = 2 * np.random.random((X.shape[1], hidden_layer_size)) - 1
    to_output_syn = 2 * np.random.random((hidden_layer_size, y.shape[1])) - 1
    syn = [0] * syn_count

    syn[0], syn[syn_count - 1] = from_input_syn, to_output_syn

    for i in xrange(1,syn_count-1):
        syn[i] = 2 * np.random.random((hidden_layer_size, hidden_layer_size)) - 1

    l = [0] * num_of_layers
    l[0] = X
    l_error = [None] * num_of_layers
    l_delta = [None] * num_of_layers

    for j in xrange(50000):
        for i in xrange(1,num_of_layers):
            l[i] = sigmoid(np.dot(l[i-1],syn[i-1]))
            if do_dropout:
                k = np.random.binomial(
                            [np.ones((l[i].shape[0],syn[i-1].shape[1]))],\
                        1 - dropout_percent)[0]
                l[i] *= k * (1.0 / (1 - dropout_percent))

        l_error[num_of_layers - 1] = l[num_of_layers - 1] - y
        if (j % 10000) == 0:
                print "Error: ", str(np.mean(np.abs(l_error[num_of_layers-1])))
        l_delta[num_of_layers - 1] = l_error[num_of_layers-1] * sigmoid_derivative(l[num_of_layers-1])

        for i in xrange(num_of_layers - 2 ,0,-1):
            l_error[i] = l_delta[i+1].dot(syn[i].T)
            l_delta[i] = l_error[i] * sigmoid_derivative(l[i])

        for i in xrange(syn_count):
            syn[i] -= alpha * l[i].T.dot(l_delta[i+1])
    print l[num_of_layers -1]
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import scipy as scipy
import wave


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1-x)

X = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

num_of_layers = 3
alpha = 30
hidden_dim = 50
dropout_percent = 0.01
do_dropout = True
y = np.array([[0, 1, 1, 0]]).T
alphas = [0.1, 1, 10, 100, 1000]
syn = [0] * num_of_layers


syn_input = 2 * np.random.random((3, hidden_dim)) - 1
syn_output = 2 * np.random.random((hidden_dim, 1)) - 1

syn[0], syn[num_of_layers-1] = syn_input, syn_output
for i in xrange(1, num_of_layers -1):
    syn[i] = 2 * np.random.random((hidden_dim,hidden_dim))

syn = [syn_input, syn_output]
l_delta = [0] * num_of_layers
l_error = [0] * num_of_layers
l = [0] * num_of_layers
l[0] = X
output_layer_index = len(syn)

for alpha in alphas:
    print str(alpha)
    for j in xrange(5000):
        for i in xrange(1, output_layer_index+1):
            l[i] = sigmoid(np.dot(l[i-1], syn[i-1]))
            if (do_dropout):
                k = np.random.binomial(
                        [np.ones((l[i-1].shape[0],syn[i-1].shape[1]))],\
                    1 - dropout_percent)[0]
                l[i] *= k * (1.0 / (1 - dropout_percent))

        l_error[output_layer_index] = l[output_layer_index] - y
        if (j % 1000) == 0:
            print "Error: ", str(np.mean(np.abs(l_error[output_layer_index])))

        l_delta[output_layer_index] = l_error[output_layer_index]\
                                        * sigmoid_derivative(l[output_layer_index])
        for i in xrange(len(syn)-1, 0, -1):
            l_error[i] = l_delta[i+1].dot(syn[i].T)
            l_delta[i] = l_error[i] * sigmoid_derivative(l[i])

        for i in xrange(len(syn)):
            syn[i] -= alpha * l[i].T.dot(l_delta[i+1])
    print l[2]

'''