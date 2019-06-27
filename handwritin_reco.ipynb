import numpy as np
import os

def loadfilename():
    def download(filename,source = 'http://yann.lecun.com/exdb/mnist/'):
        print('continue')
        print("Downloading ",filename)
        import  urllib.request
        urllib.request.urlretrieve(source+filename,filename)
    
    import gzip
    
    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename,'rb') as f:
            data = np.frombuffer(f.read(),np.uint8,offset = 16)
            data = data.reshape(-1,1,28,28)
            return data/np.float32(256)
    
    def load_mnist_labels(filename):
        if not os.path.exists(filename):            
            download(filename)
        with gzip.open(filename,'rb') as f:
            data = np.frombuffer(f.read(),np.uint8,offset = 8)
            
        return data
    
    x_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    x_test  = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test =  load_mnist_labels('t10k-labels-idx1-ubyte.gz')
    
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = loadfilename()

import matplotlib as mat
mat.use('TkAgg')

from matplotlib import pyplot as plt

plt.imshow(x_train[2][0])
plt.show()
plt.savefig('graph.png')

import lasagne
import theano
import theano.tensor as T

def built_nn(input_var = None):
    l_in = lasagne.layers.InputLayer(shape = (None,1,28,28),input_var = input_var)
    l_in_drop = lasagne.layers.DropoutLayer(l_in,p = 0.2)

    l_hid1 = lasagne.layers.DenseLayer(l_in_drop,num_units = 800,nonlinearity= lasagne.nonlinearities.rectify, W = lasagne.init.GlorotUniform())

    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1,p = 0.5)

    l_hid2 = lasagne.layers.DenseLayer(l_hid1_drop,num_units = 800,nonlinearity= lasagne.nonlinearities.rectify, W = lasagne.init.GlorotUniform())

    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2,p = 0.5)

    l_out = lasagne.layers.DenseLayer(l_hid2_drop,num_units = 10,nonlinearity= lasagne.nonlinearities.softmax)

    return l_out
print('continue2')
input_var = T.tensor4('inputs')
target_var = T.ivector('targets')

network = built_nn(input_var)
prediction  =  lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)

loss = loss.mean()
params = lasagne.layers.get_all_params(network, trainable = True)
updates = lasagne.updates.nesterov_momentum(loss,params,learning_rate = 0.1,momentum = 0.9) # for updating weights
train_fn = theano.function([input_var,target_var],loss,updates = updates)

training_steps = 10
print("Model is ready to get trained ---->")
for steps in range(training_steps):
    print("current step is : {}".format(steps))
    train_error = train_fn(x_train,y_train)
    
