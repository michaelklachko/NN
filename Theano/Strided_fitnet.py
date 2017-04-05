# -*- coding: utf-8 -*-
"""
Created on Sun Sep 04 15:46:04 2016

@author: Michael
"""


import cPickle
import time
import numpy as np
import matplotlib.pyplot as plt
from pylab import imshow, show, cm

import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.nnet import softmax
#from theano.tensor import shared_randomstreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.signal.pool import pool_2d
from theano.tensor import tanh
from theano.tensor.extra_ops import to_one_hot

#from fitter import Fitter


def ReLU(z):
    #return T.switch(z > 0, gain*z, leak*gain*z)
    #return T.maximum(gain*z, leak*gain*z) 
    return T.nnet.relu(gain*z, alpha=leak)

      
#def ReTanh(z): return T.maximum(0.0, tanh(gain*z))
def ReTanh(z): return T.maximum(0.0, tanh(gain*z))
    

class Network(object):

    def __init__(self, layers, mini_batch_size):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.

        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        
        #save initial weights:
        #save_weights(init_weights_file, self.params)
        
        self.x = T.matrix("x")
        self.y = T.ivector("y")
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        
        for j in xrange(1, len(self.layers)):
            prev_layer, layer  = self.layers[j-1], self.layers[j]
            layer.set_inpt(prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
            
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout
        

    def SGD(self, training_data, epochs, mini_batch_size, test_data, momentum, 
            timeout=0, lmbda=0.0, add_noise=[], clip_weights=[], round_down=[], early_stopping=0):
        
        global best_noise_accuracy
        global best_clipped_accuracy
        global best_round_down_accuracy
        
        start_time = time.time()
        test_results = []
        best_accuracy = 0 
        
        #load initial weights:
        #load_weights(init_weights_file, self.params)
        
        """Train the network using mini-batch stochastic gradient descent."""
        training_x, training_y = training_data
        test_x, test_y = test_data

        # compute number of minibatches for training, validation and testing
        num_training_batches = size(training_data)/mini_batch_size
        num_test_batches = size(test_data)/mini_batch_size

        # define the (regularized) cost function, symbolic gradients, and updates
        #w1,b1,w2,b2,w3,b3,w4,b4 = self.params
        #w1,b1,w2,b2 = self.params
        
        """
        #for 0.1 gain with LR=2, DR=0.3
        L2 = 15*(w1**2).sum() + 20*(b1**2).sum() + \
            1*(w2**2).sum() + 6*(b2**2).sum() +  \
              0*(w3**2).sum() + 2*(b3**2).sum() +  \
             10*(w4**2).sum() + 8*(b4**2).sum()
        """

        #L2 = sum([(layer.w**2).sum() + (layer.b**2).sum() for layer in net.layers])
        #L2 = sum([(layer.w**2).sum() for layer in net.layers])
        #l1_norm = 1*T.sum(T.abs_(w1)) #+ np.sum(np.abs(b1)) + 5*np.sum(np.abs(w2)) + 1*np.sum(np.abs(b2))
        #L1 = sum([(np.abs(layer.w)).sum() + (np.abs(layer.b)).sum() for layer in self.layers])       
        #L1 ~20 times larger than L2 
        
        cost = self.layers[-1].cost(self) #+ 0.5*lmbda*L2 #+ 0.025*lmbda*L2  #/num_training_batches  #why do we divide by num_training_batches???

        LR = theano.shared(np.asarray(LR_init, dtype=theano.config.floatX))

        if momentum:
            updates = []
            for param in self.params:
                # For each parameter, we'll create a param_update shared variable.
                # This variable will keep track of the parameter's update step across iterations.
                # We initialize it to 0
                param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
                # Each parameter is updated by taking a step in the direction of the gradient.
                # However, we also "mix in" the previous step according to the given momentum value.
                # Note that when updating param_update, we are using its old value and also the new gradient step.
                #updates.append((param, param - LR*param_update))   #replace param with updated param
                #updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param))) #replace update with m*upd_old + (1-m)*upd_new
                
                updates.append((param, param + param_update))     
                updates.append((param_update, momentum * param_update - LR * T.grad(cost, param)))
                
        else:
            grads = T.grad(cost, self.params)
            updates = [(param, param - LR*grad) for param, grad in zip(self.params, grads)]  #should we divide LR by minibatch size???

        update_learning_rate = theano.function([], LR, updates = [(LR, LR * LR_decay)])
        
        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.
        i = T.lscalar() # mini-batch index
        
        train_mb = theano.function([i], cost, updates=updates,
            givens={
                self.x: training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y: training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })

        train_mb_accuracy = theano.function([i], self.layers[-1].accuracy(self.y),
            givens={
                self.x: training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y: training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })

        test_mb_accuracy = theano.function([i], self.layers[-1].accuracy(self.y),
            givens={
                self.x: test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y: test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        self.test_mb_predictions = theano.function([i], self.layers[-1].y_out,
            givens={self.x: test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]})
        
        
        #theano.printing.debugprint(train_mb_accuracy, print_type=True)
        
        
        params_init = []
        for param in self.params:
            params_init.append(param.get_value())
            
            
        weights_saved=False
        #"""
        for k in xrange(epochs):
            """
            update_learning_rate()
            if (k+1) % 200 == 0:
                update_learning_rate()
                print "\nNew learning rate: {:.4f}".format(float(LR.get_value())), 
                if weights_saved:
                    load_weights(weights_file+str(n+10)+'.pkl', self.params)
            """
            for minibatch_index in xrange(num_training_batches):
                train_mb(minibatch_index)
            test_accuracy = 100*np.mean([test_mb_accuracy(j) for j in xrange(num_test_batches)], dtype=theano.config.floatX)
            test_results.append(test_accuracy)
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                print "\x1b[35m{:.2f}\x1b[0m".format(test_accuracy),
                if k > 100:               
                    save_weights(weights_file+str(n+10)+'.pkl', self.params)
                    weights_saved=True
                    #print "(weights saved)",
            else:
                print "{:.2f}".format(test_accuracy),
        
        if weights_saved:   #load the weights for best accuracy
            load_weights(weights_file+str(n+10)+'.pkl', self.params)
            #print "\nBest weights loaded.\n"
            
        #plot_params(self.params)
        
        test_accuracy = 100*np.mean([test_mb_accuracy(j) for j in xrange(num_test_batches)], dtype=theano.config.floatX)
        train_accuracy = 100*np.mean([train_mb_accuracy(j) for j in xrange(num_training_batches)], dtype=theano.config.floatX)           
        print "\n\nTraining Accuracy:   {:.2f}%\nTesting Accuracy:    {:.2f}%".format(train_accuracy, test_accuracy)
        print "\n--- Program ran for {:.1f} minutes, and completed {:d} epochs of training ---\n".format((time.time() - start_time)/60.0, k+1)            
        
        return test_results
        

class ConvPoolLayer(object):

    def __init__(self, a, convc, wscale, filter_shape, input_shape, pool=True, pooltype='max', poolsize=(2, 2), subsample=(1,1), pad=(0,0), p_dropout=0.0):
        """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.

        `input_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.
        """
        self.filter_shape = filter_shape
        #print "\nconv layer filter shape:", self.filter_shape
        self.input_shape = input_shape
        #print "conv layer input shape:", self.input_shape
        self.poolsize = poolsize
        self.pad = pad
        #print "\nconv layer padding:", self.pad
        self.subsample = subsample
        #print "conv layer subsample:", self.subsample
        self.pooltype = pooltype
        self.pool = pool
        self.convc = convc
        self.wscale = wscale
        self.p_dropout = p_dropout
        self.a = a

        if init=='uniform':
            fan_in = np.prod(filter_shape[1:])
            fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize))
            conv_bound = self.convc*np.sqrt(6. / (fan_in + fan_out))
            
            #print "conv_bound: {:.3f} (convc: {:d})".format(conv_bound, convc)
            self.w = theano.shared(np.asarray(np.random.uniform(low=-conv_bound, high=conv_bound, size=filter_shape), dtype=theano.config.floatX), borrow=True)                                                            

        else:
            n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
            self.w = theano.shared(np.asarray(np.random.normal(loc=0+woff, scale=np.sqrt(wscale*2.0/n_out), size=filter_shape), dtype=theano.config.floatX), borrow=True)
                    
        self.b = theano.shared(np.asarray(np.random.normal(loc=0+boff, scale=bscale*0.01, size=(filter_shape[0],)), dtype=theano.config.floatX), borrow=True)
        self.params = [self.w, self.b]
        
        #print "conv layer init weights variance: {:.3f}".format(np.sqrt(wscale*2.0/n_out))

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape(self.input_shape)
        
        self.inpt_dropout = dropout_layer(inpt_dropout.reshape(self.input_shape), self.p_dropout)
        #print "\nconv layer dropout:", self.p_dropout, '\n'

        self.conv_out = conv2d(input=self.inpt, filters=self.w, filter_shape=self.filter_shape, input_shape=self.input_shape, subsample=self.subsample)
        self.lin_output = self.a*(1-self.p_dropout)*(self.conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        #self.pos_lin_output = self.conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        #self.neg_lin_output = self.conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        self.conv_out_dropout = conv2d(input=self.inpt_dropout, filters=self.w, filter_shape=self.filter_shape, input_shape=self.input_shape, subsample=self.subsample)
        self.lin_output_dropout = self.conv_out_dropout + self.b.dimshuffle('x', 0, 'x', 'x')
        
        if self.pool:
            self.pooled_out = pool_2d(self.conv_out, ws=self.poolsize, ignore_border=True, st=None, pad=self.pad, mode=self.pooltype)
            self.pooled_out_dropout = pool_2d(self.conv_out_dropout, ws=self.poolsize, ignore_border=True, st=None, pad=self.pad, mode=self.pooltype)
            
            self.output = activation_fn(self.pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            self.output_dropout = activation_fn(self.pooled_out_dropout + self.b.dimshuffle('x', 0, 'x', 'x'))
        else:
            self.output = activation_fn(self.lin_output)
            self.output_dropout = activation_fn(self.lin_output_dropout)
            
        #self.output_dropout = self.output # no dropout in the convolutional layers


class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        #print "\nFC layer input image:", int(np.sqrt(self.n_in/64)), '\n'
        self.n_out = n_out
        #self.bits = bits
        self.p_dropout = p_dropout
        
        fc_bound = fcc*np.sqrt(6. / (self.n_in + self.n_out))
        
        #print "fc_bound: {:.3f}\n".format(fc_bound)
        
        # Initialize weights and biases
        self.w = theano.shared(np.asarray(np.random.uniform(low=-fc_bound, high=fc_bound, size=(n_in, n_out)), dtype=theano.config.floatX), name='w', borrow=True)
        #self.w = theano.shared(np.asarray(np.random.normal(loc=0.0+woff, scale=wscale*np.sqrt(2.0/n_in), size=(n_in, n_out)), dtype=theano.config.floatX), name='w', borrow=True)
        self.b = theano.shared(np.asarray(np.random.normal(loc=0.0+boff, scale=bscale*0.01, size=(n_out,)), dtype=theano.config.floatX), name='b', borrow=True)
        self.params = [self.w, self.b]
        
        #print "FC layer init weights variance: {:.3f}".format(wscale*2.0*np.sqrt(1.0/n_in))
        #print

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.inpt_dropout = dropout_layer(inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        #self.adc_input = quantize_outputs(self.inpt)
        #self.adc_input_dropout = quantize_outputs(self.inpt_dropout)
        #self.adc_output = activation_fn((1-self.p_dropout)*T.dot(self.adc_input, self.w) + self.b)
        #self.adc_output_dropout = activation_fn(T.dot(self.adc_input_dropout, self.w) + self.b)
        
        self.lin_output = (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b
        self.clean_output = activation_fn(self.lin_output)
        self.clean_output_dropout = activation_fn(T.dot(self.inpt_dropout, self.w) + self.b)

        if bits: 
            #print "\nquantized output:\n"
            self.output = self.adc_output
            self.output_dropout = self.adc_output_dropout
        else:
            self.output = self.clean_output
            self.output_dropout = self.clean_output_dropout
            #print "\nclean output:\n"
        

class OutputLayer(object):

    def __init__(self, n_in, n_out, cost_type='MSE', p_dropout=0.0):
        self.cost_type = cost_type
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        
        out_bound = outc*np.sqrt(6. / (self.n_in + self.n_out))
        
        #print "fc_bound: {:.3f}\n".format(fc_bound)
        
        # Initialize weights and biases
        self.w = theano.shared(np.asarray(np.random.uniform(low=-out_bound, high=out_bound, size=(n_in, n_out)), dtype=theano.config.floatX), name='w', borrow=True)
        #self.w = theano.shared(np.asarray(np.random.normal(loc=0.0+woff, scale=wscale*2.0*np.sqrt(1.0/n_in), size=(n_in, n_out)), dtype=theano.config.floatX), name='w', borrow=True)
        self.b = theano.shared(np.asarray(np.random.normal(loc=0.0, scale=bscale*0.01, size=(n_out,)), dtype=theano.config.floatX), name='b', borrow=True)
        
        #self.w = theano.shared(np.zeros((n_in, n_out), dtype=theano.config.floatX), name='w', borrow=True)
        #self.b = theano.shared(np.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)
        
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.inpt_dropout = dropout_layer(inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.lin_output = (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b
        
        if self.cost_type == "MSE":
            self.output = self.lin_output
            self.output_dropout = T.dot(self.inpt_dropout, self.w) + self.b 
        else:
            self.output = softmax(self.lin_output)
            self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)
            
        self.y_out = T.argmax(self.output, axis=1)

    def cost(self, net):
        if self.cost_type == "MSE":
            #both net.y and self.output have shape of (minibatch_size, n_out)
            #each row is outputs for a single image. sum axis=1 means sum values in each row, there will be n_rows results.
            return T.mean(T.sum((self.output_dropout - to_one_hot(net.y, n_out)) ** 2.0, axis=1), dtype=theano.config.floatX)  #also T.eye(n_out)[net.y]
        else:  
            #NLL cost
            #T.arange(y.shape[0]) is a vector of integers [0,1,2,...,len(y)].
            #Indexing a matrix M by the two vectors [0,1,...,K], [a,b,...,k] returns the
            #elements M[0,a], M[1,b], ..., M[K,k] as a vector.  Here, we use this
            #syntax to retrieve the log-probability of the correct labels, y.            
            return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y], dtype=theano.config.floatX)

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out), dtype=theano.config.floatX)


def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]

def print_weights(params, truncated=True):
    if truncated:
        np.set_printoptions(threshold=1000, precision=3, suppress=True) 
    else:
        np.set_printoptions(threshold=np.nan, precision=3, suppress=True) 
        
    for param in params:
        values = param.get_value()
        print values

def get_weights(params):
    count = 0
    weights = []   
    for param in params:
        weights.extend(param.get_value().flatten())
    for w in weights:
        if w != 0:
            count += 1
    return float(count)/len(weights)
    
        
def quantize_weights_slow(params, num_bits): #rounding with a loop
    levels = []       
    num_bins = 2**num_bits
    bin_size = range_ / num_bins
    half_bin = bin_size/2
    for i in xrange(num_bins):
        levels.append(min_ + (i+0.5)*bin_size)
    #print "Quantization thresholds:\n", levels
    #print "\nPrecision =", num_bits, "bits:\n" 
        
    for param in params:
        values = param.get_value()
        values_q = quantize_slow_vec(values, half_bin, levels)
        values_q = values_q.astype(np.float32, copy=False)
        param.set_value(values_q)
            
            
def quantize_slow(value, half_bin, levels):    
    for level in levels:
        if abs(level-value) <= half_bin:
            return level
    if value < min_: return levels[0]
    if value > max_: return levels[-1]

quantize_slow_vec = np.vectorize(quantize_slow, excluded=[1,2])

def quantize_weights(params, bits):
    for param in params:
        values = param.get_value()
        values = quantize_vec(values, bits)
        values = values.astype(np.float32, copy=False)
        param.set_value(values)

def cutoff(a, p):
    """given the percentage p, returns p% largest values of a, or p% smallest values of a (default is 'smallest')"""
    
    b = a.flatten()
    c = np.sort(b)
    low = c[int(p*len(c))]     #highest of the first p% of elements in b
    high = c[-int(p*len(c))]   #lowest of the last p% of elements in b
    
    b[b<low] = low
    b[b>high] = high
    
    return b.reshape(a.shape)

                
                
        
def adc(net):
    #####    Quantized Input to the FC layer    #######
    
    np.set_printoptions(threshold=np.nan, precision=3, suppress=True)
    
    a = training_data[0].get_value()[0:100].reshape(100,3,32,32)
    l0_out = theano.function([net.layers[0].inpt], net.layers[0].output)
    l1_out = theano.function([net.layers[0].output], net.layers[1].output)
    fc_in = l1_out(l0_out(a)).reshape((100, 64*25))
    adc_in = theano.function([net.layers[2].inpt_reshaped], net.layers[2].adc_inpt)
    adc_in(fc_in)[0][:40]
    quantize_outputs(fc_in, 8)[0][:40].eval()
    dropout_layer(fc_in, 0.3)[0][:200].eval()


def ff_gpu(net, plot_o=True, plot_w=True, plot_z=True, plot_wsums=True, plot_norms=True, plot_diff=True):  #plot=['o', 'w', 'z', 'wsums', 'norms', diffs']):

    ######   GPU code for Convolutional Network    ######
    
    start_time = time.time()
    
    params = []
    for param in net.params:
        params.append(param.get_value())
    
    shared_params = []
    for p in params:
        pos_p = np.clip(p, 0, np.max(p))
        pos_shared = theano.shared(np.asarray(pos_p, dtype=theano.config.floatX), borrow=True)
        neg_p = np.clip(p, np.min(p), 0)
        neg_shared = theano.shared(np.asarray(neg_p, dtype=theano.config.floatX), borrow=True)
        shared_params.append((pos_shared, neg_shared))

    #shared_params = [(wp1, wn1), (bp1, bp1), (wp2, wn2), (bp2, bp2), (wp3, wn3), (bp3, bp3), (wp4, wn4), (bp4, bp4)]    
    
    a0 = training_data[0].get_value()[0:num_img].reshape(num_img,fm0,img_in,img_in)
    
    z = []
    zp = []
    zn = []
    outputs = [a0]        
        
    l0_lin_out = theano.function([net.layers[0].inpt], net.layers[0].lin_output)
    l0_pos_lin_out = theano.function([net.layers[0].inpt], net.layers[0].lin_output, 
                    givens={net.params[0]: shared_params[0][0], net.params[1]: shared_params[1][0]})
    l0_neg_lin_out = theano.function([net.layers[0].inpt], net.layers[0].lin_output, 
                    givens={net.params[0]: shared_params[0][1], net.params[1]: shared_params[1][1]})
    l0_out = theano.function([net.layers[0].inpt], net.layers[0].output)
    
    z1 = l0_lin_out(a0)
    zp1 = l0_pos_lin_out(a0) 
    zn1 = l0_neg_lin_out(a0)
    a1 = l0_out(a0)
    
    z.append(z1)
    zp.append(zp1)
    zn.append(zn1)
    outputs.append(a1)
    
    for i in range(1,4):
        lin_out = theano.function([net.layers[i-1].output], net.layers[i].lin_output)
        pos_lin_out = theano.function([net.layers[i-1].output], net.layers[i].lin_output, 
                        givens={net.params[2*i]: shared_params[2*i][0], net.params[2*i+1]: shared_params[2*i+1][0]})
        neg_lin_out = theano.function([net.layers[i-1].output], net.layers[i].lin_output, 
                        givens={net.params[2*i]: shared_params[2*i][1], net.params[2*i+1]: shared_params[2*i+1][1]})
        out = theano.function([net.layers[i-1].output], net.layers[i].output)        

        z.append(lin_out(outputs[-1]))
        zp.append(pos_lin_out(outputs[-1])) 
        zn.append(neg_lin_out(outputs[-1])) 
        
        a = out(outputs[-1])
        outputs.append(a)
    
    #a0, a1, a2, a3, a4 = outputs
    
        
    print "\n--- Program ran for {:.1f} seconds ---\n".format(time.time() - start_time)    
        
    """    
    w1 = net.params[0].get_value()
    w2 = net.params[2].get_value()
    w3 = net.params[4].get_value()
    w4 = net.params[6].get_value()
    
    b1 = net.params[1].get_value()
    b2 = net.params[3].get_value()
    b3 = net.params[5].get_value()
    b4 = net.params[7].get_value()
    
    bp1 = np.clip(b1, 0, np.max(b1)) 
    bn1 = np.clip(b1, np.min(b1), 0) 
    wp1 = np.clip(w1, 0, np.max(w1))
    wn1 = np.clip(w1, np.min(w1), 0)
    
    bp2 = np.clip(b2, 0, np.max(b2)) 
    bn2 = np.clip(b2, np.min(b2), 0) 
    wp2 = np.clip(w2, 0, np.max(w2))
    wn2 = np.clip(w2, np.min(w2), 0)
    
    bp3 = np.clip(b3, 0, np.max(b3)) 
    bn3 = np.clip(b3, np.min(b3), 0) 
    wp3 = np.clip(w3, 0, np.max(w3))
    wn3 = np.clip(w3, np.min(w3), 0)
    
    bp4 = np.clip(b4, 0, np.max(b4)) 
    bn4 = np.clip(b4, np.min(b4), 0) 
    wp4 = np.clip(w4, 0, np.max(w4))
    wn4 = np.clip(w4, np.min(w4), 0)
    
    shared_bp1 = theano.shared(np.asarray(bp1, dtype=theano.config.floatX), borrow=True)
    shared_bn1 = theano.shared(np.asarray(bn1, dtype=theano.config.floatX), borrow=True)
    shared_wp1 = theano.shared(np.asarray(wp1, dtype=theano.config.floatX), borrow=True)
    shared_wn1 = theano.shared(np.asarray(wn1, dtype=theano.config.floatX), borrow=True)
    
    shared_bp2 = theano.shared(np.asarray(bp2, dtype=theano.config.floatX), borrow=True)
    shared_bn2 = theano.shared(np.asarray(bn2, dtype=theano.config.floatX), borrow=True)
    shared_wp2 = theano.shared(np.asarray(wp2, dtype=theano.config.floatX), borrow=True)
    shared_wn2 = theano.shared(np.asarray(wn2, dtype=theano.config.floatX), borrow=True)
    
    shared_bp3 = theano.shared(np.asarray(bp3, dtype=theano.config.floatX), borrow=True)
    shared_bn3 = theano.shared(np.asarray(bn3, dtype=theano.config.floatX), borrow=True)
    shared_wp3 = theano.shared(np.asarray(wp3, dtype=theano.config.floatX), borrow=True)
    shared_wn3 = theano.shared(np.asarray(wn3, dtype=theano.config.floatX), borrow=True)
    
    shared_bp4 = theano.shared(np.asarray(bp4, dtype=theano.config.floatX), borrow=True)
    shared_bn4 = theano.shared(np.asarray(bn4, dtype=theano.config.floatX), borrow=True)
    shared_wp4 = theano.shared(np.asarray(wp4, dtype=theano.config.floatX), borrow=True)
    shared_wn4 = theano.shared(np.asarray(wn4, dtype=theano.config.floatX), borrow=True)
       
        
    a0 = training_data[0].get_value()[0:num_img].reshape(num_img,fm0,img_in,img_in)   
    l0_lin_out = theano.function([net.layers[0].inpt], net.layers[0].lin_output)
    l0_pos_lin_out = theano.function([net.layers[0].inpt], net.layers[0].lin_output, 
                    givens={net.params[0]: shared_wp1, net.params[1]: shared_bp1})
    l0_neg_lin_out = theano.function([net.layers[0].inpt], net.layers[0].lin_output, 
                    givens={net.params[0]: shared_wn1, net.params[1]: shared_bn1})
    l0_out = theano.function([net.layers[0].inpt], net.layers[0].output)
    z1 = l0_lin_out(a0)
    zp1 = l0_pos_lin_out(a0) 
    zn1 = l0_neg_lin_out(a0)
    
    a1 = l0_out(a0)
    l1_lin_out = theano.function([net.layers[0].output], net.layers[1].lin_output)
    l1_pos_lin_out = theano.function([net.layers[0].output], net.layers[1].lin_output, 
                    givens={net.params[2]: shared_wp2, net.params[3]: shared_bp2})
    l1_neg_lin_out = theano.function([net.layers[0].output], net.layers[1].lin_output, 
                    givens={net.params[2]: shared_wn2, net.params[3]: shared_bn2})
    l1_out = theano.function([net.layers[0].output], net.layers[1].output)
    z2 = l1_lin_out(a1)
    zp2 = l1_pos_lin_out(a1) 
    zn2 = l1_neg_lin_out(a1)
    
    a2 = l1_out(a1)
    #a2 = a2_init.reshape((num_img, fm2*img4*img4))
    l2_lin_out = theano.function([net.layers[1].output], net.layers[2].lin_output)
    l2_pos_lin_out = theano.function([net.layers[1].output], net.layers[2].lin_output, 
                    givens={net.params[4]: shared_wp3, net.params[5]: shared_bp3})
    l2_neg_lin_out = theano.function([net.layers[1].output], net.layers[2].lin_output, 
                    givens={net.params[4]: shared_wn3, net.params[5]: shared_bn3})
    l2_out = theano.function([net.layers[1].output], net.layers[2].output)
    z3 = l2_lin_out(a2)
    zp3 = l2_pos_lin_out(a2) 
    zn3 = l2_neg_lin_out(a2)
    
    a3 = l2_out(a2)
    l3_lin_out = theano.function([net.layers[2].output], net.layers[3].lin_output)
    l3_pos_lin_out = theano.function([net.layers[2].output], net.layers[3].lin_output, 
                    givens={net.params[6]: shared_wp4, net.params[7]: shared_bp4})
    l3_neg_lin_out = theano.function([net.layers[2].output], net.layers[3].lin_output, 
                    givens={net.params[6]: shared_wn4, net.params[7]: shared_bn4})
    l3_out = theano.function([net.layers[2].output], net.layers[3].lin_output)
    z4 = l3_lin_out(a3)
    zp4 = l3_pos_lin_out(a3) 
    zn4 = l3_neg_lin_out(a3)
    a4 = l3_out(a3)
    
    print "\n--- Program ran for {:.1f} seconds ---\n".format(time.time() - start_time)
    
    outputs = [a0, a1, a2, a3, a4]
    weights = [w1, w2, w3, w4]
    z = [z1, z2, z3, z4]
    zp = [zp1, zp2, zp3, zp4]
    zn = [zn1, zn2, zn3, zn4]
    #"""
    
    def plot_outs(a):
        for i, a in enumerate(outputs):
            plt.hist(a.flatten(), bins=270)
            plt.title("Layer {:d} outputs, {:d} images".format(i, num_img), fontsize=16)
            plt.yscale('log')
            plt.show()
    
    def plot_ws(layer, values, bins=200):
        plt.hist(values, bins=bins)
        plt.title("Layer {:d} weights".format(layer), fontsize=16)
        #plt.yscale('log')
        plt.show()
            
    def plot_wall(weights):
        for i, w in enumerate(weights):
            plot_ws(i+1, w.flatten())
    
    def plot_wsum(layer, values, bins=200):
        plt.hist(values, bins=bins, label="max: {:.2f}\nmin: {:.2f}".format(np.max(values), np.min(values)))
        plt.title("Layer {:d} sum of output weights".format(layer), fontsize=16)
        plt.legend(loc='upper right')
        #plt.yscale('log')
        plt.show()
        
    def plot_norm_avg(values, label='', title='', bins=70):
        plt.hist(values, bins=70, label=label)
        plt.title(title, fontsize=12)
        plt.legend(loc='upper right')
        plt.show()
    
    def plot_zs(z, zp, zn):
        count = 1
        for t, p, n in zip(z, zp, zn):
            plot_output(num_img, count, t.flatten())
            plot_outputs_compare(num_img, count, p.flatten(), n.flatten())
            count += 1      
    
    
    if plot_o:
        plot_outs(outputs)
    if plot_w:
        weights = [params[i] for i in range(0,len(params), 2)] 
        plot_wall(weights)
    if plot_z:
        plot_zs(z, zp, zn)    

    if plot_wsums:
        weights = [params[i] for i in range(0,len(params), 2)] 
        plot_wsum(1, np.sum(np.abs(weights[0]), axis=(0,2,3)), bins=100)   
        plot_wsum(2, np.sum(np.abs(weights[1]), axis=(0,2,3)), bins=100)
        plot_wsum(3, np.sum(np.abs(weights[2].transpose()), axis=0), bins=100)
        plot_wsum(4, np.sum(np.abs(weights[3].transpose()), axis=0), bins=100)
        
    if plot_norms:
        #sum of abs. values of positive and negative inputs, per layer
        zt = [p - n for p,n in zip(zp, zn)]
        zt1, zt2, zt3, zt4 = zt
        
        #average layer input per image
        zx1 = np.mean(zt1.reshape(num_img, -1), axis=1)/(fm0*fs1*fs1)
        zx2 = np.mean(zt2.reshape(num_img, -1), axis=1)/(fm1*fs1*fs1)
        zx3 = np.mean(zt3, axis=1)/(fm2*fs2*fs2)
        zx4 = np.mean(zt4, axis=1)/fc
         
        #zs: sum of pre-act. inputs in a layer, for a single image.  
        zs1 = np.sum(zt1.reshape(num_img, -1), axis=1)
        zs2 = np.sum(zt2.reshape(num_img, -1), axis=1)
        zs3 = np.sum(zt3, axis=1)
        zs4 = np.sum(zt4, axis=1)
        
        #tt: sum of pre-act. inputs in all layers, for a single image.
        tt = zs1+zs2+zs3+zs4
        
        #ss: N of neurons * Theoretical Maximum of pre-act inputs to each neuron
        ss = n_out*fc + fc*(fm2*fs2*fs2) + (fm2*img4*img4)*(fm1*fs1*fs1) + (fm1*img2*img2)*(fm0*fs1*fs1)
        
        plot_norm_avg(zx1, label='Convnet\nLayer 1\nCIFAR', title='Normalized averages of pre-activation inputs per image (10k images)', bins=70)
        plot_norm_avg(zx2, label='Convnet\nLayer 2\nCIFAR', title='Normalized averages of pre-activation inputs per image (10k images)', bins=70)
        plot_norm_avg(zx3, label='Convnet\nLayer 3\nCIFAR', title='Normalized averages of pre-activation inputs per image (10k images)', bins=70)
        plot_norm_avg(zx4, label='Convnet\nLayer 4\nCIFAR', title='Normalized averages of pre-activation inputs per image (10k images)', bins=70)
        plot_norm_avg(tt/ss, label='Convnet\nTotal\nCIFAR', title='Normalized averages of pre-activation inputs per image (10k images)', bins=70)
        
    if plot_diff:
        m = np.max(z[-1], axis=1)
        mm = np.argmax(z[-1], axis=1)
        mmm = np.eye(10)[mm]
        mb = mmm.astype(bool)
        z4m = np.copy(z[-1])
        z4m[mb] = -10000
        s = np.max(z4m, axis=1)
        res = m - s
        v = res/np.max(res)
        
        plt.hist(v, label="max: {:.2f}\nmin: {:.6f}".format(np.max(res), np.min(res)), bins=100)
        plt.title("Difference between 1st and 2nd outputs, {:d} images".format(num_img), fontsize=16)  #Difference between highest and second highest outputs\nper image, 10k images, 
        plt.legend(loc='upper right')
        plt.show()
        
        plt.hist(v[v<np.max(res)/100.], bins=20)
        plt.title("Leftmost bar, {:d} images".format(len(v[v<np.max(res)/100.])), fontsize=16)
        #plt.legend(loc='upper right')
        plt.show()

    #return z, zp, zn
    
    
def quantize_outputs(layer):
    #print "\n\nbits =", bits, '\n\n'
    ff = T.pow(2.0, bits) - 1    #must use 2.0 base, 2 will cast the output to int8!!!!
    #print "\nlevels:", ff.eval()
    #print "\nexample value:", (T.round(0.567*ff)/ff).eval()
    return T.round(layer*ff)/ff
    #return T.iround(layer*levels)/levels
        
def quantize(value, bits): 
    #print "\n\nbits =", bits, '\n\n'
    offset = 2**(bits-1)    
    if value >= 0:
        return (value+offset >> bits) << bits
    else:
        return ((value-offset >> bits) << bits) + offset*2

quantize_vec = np.vectorize(quantize, excluded=[1])


def round_down_fun(params, threshold=0.1):     
    for param in params:
        values = param.get_value()
        values = round_down_vec(values, threshold)
        values = values.astype(np.float32, copy=False)
        param.set_value(values)            
            
def round_down(value, threshold):    
    if abs(value) <= threshold:
            return 0.0
    else:
        return value
    
round_down_vec = np.vectorize(round_down, excluded=[1])


def add_noise_fun(params, magnitude=0.1):
    for i, param in enumerate(params):
        values = param.get_value()
        values = add_noise_vec(values, magnitude)
        values = values.astype(np.float32, copy=False)
        param.set_value(values)
    
def noise(value, magnitude):
    sigma = np.abs(magnitude*value)  #sigma is proportional to the weight value, magnitude is the proportionality (eg m=0.1 means 10% of W) 
    return value + np.random.normal(0.0, sigma+0.00001)   #have to add a small number because sigma cannot be zero
    
add_noise_vec = np.vectorize(noise, excluded=[1]) 

def noise2(value, magnitude):   #faster than noise when there are lots of zeros in the array, but not by much
    
    sigma = np.abs(magnitude*value)  #noise*W, where noise is how much noise to add 
    if sigma == 0:
        return value
    else:
        return value + np.random.normal(0.0, sigma)   #have to add a small number because sigma cannot be zero
    
add_noise_vec2 = np.vectorize(noise2, excluded=[1]) 

def clip_weights_fun(params, bounds=[]):
    w_bounds = bounds
    for i, param in enumerate(params):
        values = param.get_value()
        #if i % 2 == 0:
        if i == 0:
            bounds = w_bounds
        else:
            bounds = [-100, 100] #don't clip biases
        z = np.zeros_like(values)
        low = z + bounds[0]
        high = z + bounds[1]
        print "clipping layer", i, "weights to", bounds
        values = np.minimum(np.maximum(values, low), high)
        values = values.astype(np.float32, copy=False)
        param.set_value(values)
        
def plot_weights(weights, bins=20):
    for i, layer in enumerate(weights):
        values = layer.get_value().flatten()
        if i % 2 == 0:
            #print values
            title = "Layer {} Weights:".format(i/2)
            plt.title(title, fontsize=16)
        else:
            title = "Layer {} Biases:".format(i/2)
            plt.title(title, fontsize=16)
        plt.hist(values, bins=bins)
        #plt.legend(loc='upper right')
        plt.show()
        
def plot_params(params, bins=200):
    for i in range(0,len(params),2):
        w = params[i].get_value().flatten()
        b = params[i+1].get_value().flatten()
        fig = plt.figure(figsize=(14,4))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.hist(w, bins=bins)
        ax2.hist(b, bins=bins)
        ax1.set_title("Layer {} Weights".format(i/2+1))
        ax2.set_title("Layer {} Biases".format(i/2+1))
        plt.show()

        
def plot_weights_compare_old(weights1, weights2, bins=20):
    count = 0
    for layer1, layer2 in zip(weights1, weights2):
        values1 = layer1.get_values()
        values2 = layer2.get_values()
        count += 1
        plt.hist(values1, alpha=0.5, bins=bins, color= 'g', label='after')
        plt.hist(values2, alpha=0.5, bins=bins, color= 'r', label='before')
        if count % 2 == 0:
            title = "Layer {} Biases:".format(count/2)
            plt.title(title, fontsize=16)
        else:
            title = "Layer {} Weights:".format((count+1)/2)
            plt.title(title, fontsize=16)
        plt.title("Weights")
        #plt.xlabel('Value', fontsize=16)
        #plt.ylabel('Frequency', fontsize=16)
        plt.legend(loc='upper right')
        plt.show()
        
def plot_weights_compare(weights1, weights2, bins=20):
    for i in [0,2,4,6]: 
            w_init = weights1[i].flatten()
            w_final = weights2[i].get_value().flatten()  
            b_init = weights1[i+1].flatten()
            b_final = weights2[i+1].get_value().flatten() 
            
            fig = plt.figure(figsize=(14,4))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            
            ax1.hist(w_final, alpha=0.5, bins=bins, color= 'r', label='after')
            ax1.hist(w_init, alpha=0.5, bins=bins, color= 'b', label='before')
            ax2.hist(b_final, alpha=0.5, bins=bins, color= 'r', label='after')
            ax2.hist(b_init, alpha=0.5, bins=bins, color= 'b', label='before')
            
            
            ax1.set_title("Layer {} Weights".format(i/2+1))
            ax2.set_title("Layer {} Biases:".format(i/2+1))
                
            plt.legend(loc='upper right')
            plt.show()

def plot_clipped_weights(params_orig, params_clipped, bins=300):
    count = 0
    for p_orig, p_clipped in zip(params_orig, params_clipped):
        if count % 2 == 0:
            p_clipped = p_clipped.get_value().flatten()
            fig = plt.figure(figsize=(14,4))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            ax1.hist(p_orig.flatten(), bins=bins)
            ax2.hist(p_clipped, bins=bins)
            if count % 2 == 0:
                ax1.set_title("Layer {} Weights (Before):".format(count/2+1))
                ax2.set_title("Layer {} Weights (After):".format(count/2+1))
            else:
                ax1.set_title("Layer {} Biases: (Before)".format(count/2+1))
                ax2.set_title("Layer {} Biases: (After)".format(count/2+1))
            #plt.legend(loc='upper right')
            plt.show()
        count += 1

def plot_outputs_compare(num_img, layer, values1, values2, bins=200):
    plt.hist(values1, alpha=0.5, bins=bins, color= 'g', label="max: {:.2f}".format(np.max(values1)))
    plt.hist(values2, alpha=0.5, bins=bins, color= 'r', label="min: {:.2f}".format(np.min(values2)))
    plt.title("Layer {:d} pre-activation inputs, {:d} images".format(layer, num_img), fontsize=16)
    #plt.xlabel('Value')
    #plt.ylabel('Frequency')
    #plt.semilogy()
    #ax.set_yscale('log')
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.show()
    
def plot_output(num_img, layer, values, bins=200):
    plt.hist(values, bins=bins, label="max: {:.2f}\nmin: {:.2f}".format(np.max(values), np.min(values)))
    plt.title("Layer {:d} pre-activation inputs, {:d} images".format(layer, num_img), fontsize=16)
    #plt.xlabel('Value')
    #plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    #plt.semilogx()
    #plt.semilogy()
    #ax.set_yscale('log')
    plt.yscale('log')
    plt.show()

def plot_wsum(layer, values, bins=200):
    plt.hist(values, bins=bins, label="max: {:.2f}\nmin: {:.2f}".format(np.max(values), np.min(values)))
    plt.title("Layer {:d} sum of output weights".format(layer), fontsize=16)
    #plt.xlabel('Value')
    #plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    #plt.semilogx()
    #plt.semilogy()
    #ax.set_yscale('log')
    plt.yscale('log')
    plt.show()
       
        
def dropout_layer(layer, p_dropout):
    #if p_dropout == 0:
        #return layer
    #print "\nentering dropout layer...\n"
    srng = RandomStreams(np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)
        
     
def save_weights(weights_file, params):
    values = []
    for param in params:
        values.append(param.get_value())
    f = open(weights_file, 'wb')
    cPickle.dump(values, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    
def load_weights(weights_file, params):
    f = open(weights_file, 'rb')
    weights = cPickle.load(f)
    #print weights[0].shape
    f.close()
    for w, param in zip(weights, params):
        param.set_value(w)

def load_cifar100(input_path):   
    print "\nLoading CIFAR-100..."
    f1 = open(input_path + 'train', 'rb')
    f2 = open(input_path + 'test', 'rb')
    
    tr = cPickle.load(f1)
    te = cPickle.load(f2)

    train_images = np.asarray(tr["data"], dtype=np.float32) / 255.0
    test_images = np.asarray(te["data"], dtype=np.float32) / 255.0 
    
    train_labels = tr["coarse_labels"]
    test_labels = te["coarse_labels"]
    
    #validation_data = (train_images[48000:], train_labels[48000:])
    #training_data = (train_images[:48000], train_labels[:48000])
    training_data = (train_images, train_labels)
    test_data = (test_images, test_labels)
    
    def shared(data):
        shared_x = theano.shared(np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
        
    #return [shared(training_data), shared(validation_data), shared(test_data)] 
    return [shared(training_data), shared(test_data)] 


def load_data_shared(filename):
    f = open(filename, 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    #training_data, test_data = cPickle.load(f)
    #train = [[], []]
    #train[0] = np.concatenate([training_data[0], validation_data[0]])
    #train[1] = np.concatenate([training_data[1], validation_data[1]])
    f.close()
    
    def shared(data):
        shared_x = theano.shared(np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
        
    return [shared(training_data), shared(test_data)]
    
    
def plot_total(y_range, epochs, results, variables, var_name):
    print "\n\n\n"
    train_time=range(epochs)        
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111) 
    for result, var in zip(results, variables): 
        ax.plot(train_time, result, marker='.', linestyle='-', linewidth=2.0, label=var_name+'='+str(var))
    plt.xlabel('Training Time, epochs', fontsize=16)
    plt.ylabel('Accuracy, %', fontsize=16)
    ax.set_ylim(y_range)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax.legend(loc='lower right', prop={'size':14})
    plt.show()
    print "\n\n\n"

def view_image(image):
    if colors == 1:
        imshow(np.reshape(image, (img_in,img_in)), cmap=cm.gray)
    else:
        imshow(np.reshape(image, (img_in,img_in,colors), order='F'))
    show()
    #for cifar RGB, do this: image..transpose(1,2,0)

def combine_runs():
    weights = []
    for i in range(110):
        f = open(weights_file+str(i)+'.pkl', 'rb')
        params = cPickle.load(f)
        f.close()
        weights.append(params)
        
    w1 = []
    b1 = []
    w2 = []
    b2 = []
    w3 = []
    b3 = []
    w4 = []
    b4 = []
    for sett in weights:
        w1 = np.append(w1, sett[0].flatten())
        b1 = np.append(b1, sett[1].flatten())
        w2 = np.append(w2, sett[2].flatten())
        b2 = np.append(b2, sett[3].flatten())
        w3 = np.append(w3, sett[4].flatten())
        b3 = np.append(b3, sett[5].flatten())
        w4 = np.append(w4, sett[6].flatten())
        b4 = np.append(b4, sett[7].flatten())
    return [w1,b1,w2,b2,w3,b3,w4,b4]
 
def plot_weightss(params):   
    for i in range(0,len(params),2):
        fig = plt.figure(figsize=(14,4))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.hist(params[i], bins=200)
        ax2.hist(params[i+1], bins=100)
        ax1.set_title("Layer {} Weights".format(i/2+1))
        ax2.set_title("Layer {} Biases".format(i/2+1))
        plt.show()


def fit_data(data):
    f = Fitter(data)
    f.fit()
    # may take some time since by default, all distributions are tried
    # but you can manually provide a smaller set of distributions
    f.summary()
    

def plot_fun(): 
    
    def ReLU(z): return np.maximum(0.0, gain*z)  
    def ReTanh(z): return np.maximum(0.0, np.tanh(gain*z))
        
    x = np.asarray([0.03*i for i in range(100)])
    y = ReTanh(x)
    z = ReLU(x)
       
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111) 
    
    ax.plot(x, y, marker='o', linestyle='-', color='b', linewidth=2.0, label='ReTanh')
    ax.plot(x, z, marker='o', linestyle='-', color='g', linewidth=2.0, label='ReLU')
    #ax.set_xlim([0,300])
    ax.legend(loc='center right', prop={'size':14})
    plt.show()


results = []
best_results = []
avg_results = []


cifar_RGB_0bit = "F:\Data\cifar\cifar_RGB_0bit"
cifar_RGB_1bit = "C:\Users\Michael\Desktop\Research\Data\cifar\cifar_RGB_1bit"
cifar_RGB_2bit = "C:\Users\Michael\Desktop\Research\Data\cifar\cifar_RGB_2bit"
cifar_RGB_3bit = "C:\Users\Michael\Desktop\Research\Data\cifar\cifar_RGB_3bit"
cifar_RGB_4bit = "C:\Users\Michael\Desktop\Research\Data\cifar\cifar_RGB_4bit" 
cifar_RGB_5bit = "C:\Users\Michael\Desktop\Research\Data\cifar\cifar_RGB_5bit"
cifar_RGB_6bit = "C:\Users\Michael\Desktop\Research\Data\cifar\cifar_RGB_6bit"

mnist = "C:\Users\Michael\Desktop\Research\Data\mnist\mnist.pkl"#_binary.pkl"
mnist_14 = "C:\Users\Michael\Desktop\Research\Data\mnist\mnist_14_binary.pkl"

dataset=cifar_RGB_0bit
#dataset = mnist
training_data, test_data = load_data_shared(dataset)
    
weights_file = 'C:\Users\Michael\Desktop\Research\Results\\best_weights'
clipped_weights_file = 'C:\Users\Michael\Desktop\Research\Results\\clipped_weights.pkl'
noisy_weights_file = 'C:\Users\Michael\Desktop\Research\Results\\noisy_weights_file'
round_down_weights_file = 'C:\Users\Michael\Desktop\Research\Results\\round_down_weights_file'
init_weights_file = 'C:\Users\Michael\Desktop\Research\Results\weights_init.pkl'

results_conv_0L_file = "C:\Users\Michael\Desktop\\results_conv_0L.txt"
results_conv_1L_file = "C:\Users\Michael\Desktop\\results_conv_1L.txt"

n_out=10  
timeout=180   #abort the program after this many minutes

if dataset == mnist:
    img_in=28
    colors=1
else:
    colors=3
    img_in = 32 


pooltype = 'average_exc_pad'#'max'

#cost_type = 'MSE'
cost_type = 'NLL'

gain = 0.1    #slope of activation function
activation_fn = ReLU#Tanh

if activation_fn == ReLU:
    LR_init = 0.2
else:
    LR_init = 0.4 

LR_decay = 0.2
    
epochs, mini_batch_size = 500, 128
num_img = mini_batch_size

#fm0, fm1, fm2, fc = colors, 26, 26, 300     #for MNIST: 99.61%
fm0, fm1, fm2, fc = colors, 64, 128, 390

lmbda, p_dropout, momentum = 0.00003, 0.75, 0.9
p_dropout_conv = 0.05
p_dropout_input = 0.0

lbias = 1
wscale1 = 10
wscale2 = 1
bscale = 1
bits = 0
woff = 0#0.01
boff = 0.05

init = 'normal'

convc1 = 20
convc2 = 2
fcc = 2
outc = 40

leak = 0
a1 = 1.
a2 = 1.

num_sims = 3

fs1, fs2 = 6, 6
pad1, pad2, pad3 = 0,0,1 

img1 = (img_in-fs1)/2 + 1
img2 = (img1-fs2)/2 + 1

subsample=(2, 2)  #stride of conv. filt

momentums = [0.0, 0.8, 0.9, 0.95, 0.99, 0.995]
L2s = [0.000005, 0.00001, 0.000015, 0.00002, 0.000025]
wscales = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
boffs = [-0.01, 0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
fms = [10,12,14,16,18,20,22,24,26,28,30,32,48]
fcs = [80, 100, 120, 140, 160, 180, 200, 220, 250, 300, 400, 500, 700, 1000]
leaks = [0]#0.05, 0.1, 0.15, 0.2, 0.25]
LRs = [0.02, 0.05, 0.1, 0.4, 1.]
dropouts = [0.75, 0.8, 0.85]
dropout_inputs = [0.0, 0.005, 0.01, 0.02]#, 0.15, 0.2, 0.25]
dropout_convs = [0.0, 0.05, 0.1, 0.15]#0.2, 0.3, 0.4, 0.5]
a1s = [1.2, 1.3, 1.5]#,0.9, 1.0, 1.1]

#np.set_printoptions(threshold=10000, precision=4, suppress=True)

print "\n\n{}".format(dataset)
print "\nNetwork architecture: two conv layers, one fully connected layer: {:d}-{:d}-{:d}-{:d}, filter sizes=({:d},{:d}), \
activation_fn = {} with gain={:.2f}, pooltype: {}, cost: {}".format(fm1, fm2, fc, n_out, fs1, fs2, activation_fn.__name__, gain, pooltype, cost_type)
print "\nParameters: Initial learning rate={:.2f}, minibatch size={:d}, dropout = {:.2f}, L2 lambda={:.6f}, momentum={:.2f}, initialization: {}\n\n\
Training for {:d} epochs\n".format(LR_init, mini_batch_size, p_dropout, lmbda, momentum, init, epochs)

for h in [1]:
#for i in range(7):
#for fm1 in fms:
#for fc in fcs:
#for wscale1 in wscales:
#for boff_fc in boffs:
#for lmbda in L2s:
#for leak in leaks:
#for momentum in momentums:
#for LR in LRs:
#for p_dropout in dropouts:
#for p_dropout_input in dropout_inputs:
#for p_dropout_conv in dropout_convs:
#for a1 in a1s:

    #print "fc:", fc, ' dropout:', p_dropout, '\n'
    #print "fm1:", fm1, '\n'
    #print "\nboff:", boff, '\n'
    #print "\nwscale1:", wscale1, '\n'
    #print "\nL2:", lmbda
    #print "\nMomentum:", momentum
    #print "\nleak:", leak, '\n'
    #print "\nboff_fc:", boff_fc, '\n'
    #print "\nLR:", LR, "\n"
    #print "\n\ndropout:", p_dropout, "\n"
    #print "\n\ndropout_input:", p_dropout_input, "\n"
    #print "\n\ndropout_conv:", p_dropout_conv, "\n"
    #print "\na1:", a1, "\n"
    
    """
    dataset = "C:\Users\Michael\Desktop\Research\Data\cifar\cifar_RGB_" + str(i) + "bit"
    training_data, test_data = load_data_shared(dataset)
    print
    print dataset[-14:]
    print
    """
    best_results_per_config = []
    
    for n in range(num_sims):
        
            #print "\nSimulation:", n, '\n'
            
            net = Network([
                ConvPoolLayer(a1, convc1, wscale1, input_shape=(mini_batch_size, colors, img_in, img_in), filter_shape=(fm1, colors, fs1, fs1), 
                              pool=False, pad=(pad1,pad1), pooltype=pooltype, poolsize=(2, 2), subsample=subsample, p_dropout=p_dropout_input),
                #ConvPoolLayer(input_shape=(mini_batch_size, fm1, img2, img2), filter_shape=(fm2, fm1, fs2, fs2), pad=(pad2,pad2), pooltype=pooltype, poolsize=(2, 2), subsample=subsample),
                ConvPoolLayer(a2, convc2, wscale2, input_shape=(mini_batch_size, fm1, img1, img1), filter_shape=(fm2, fm1, fs2, fs2), 
                              pool=False, pad=(pad2,pad2), pooltype=pooltype, poolsize=(2, 2), subsample=subsample, p_dropout=p_dropout_conv),
                #ConvPoolLayer(input_shape=(mini_batch_size, fm2, img4, img4), filter_shape=(fm3, fm2, fs3, fs3), pad=(pad2,pad2), poolsize=(2, 2)),
                FullyConnectedLayer(n_in=fm2*img2*img2, n_out=fc, p_dropout=p_dropout),
                #FullyConnectedLayer(n_in=fm2*img2*img2, n_out=fc, p_dropout=p_dropout),
                ####FullyConnectedLayer(n_in=img_in*img_in*colors, n_out=fc, p_dropout=p_dropout),
                #FullyConnectedLayer(n_in=lsize, n_out=lsize, p_dropout=p_dropout),
                #FullyConnectedLayer(n_in=fc, n_out=lsize2, p_dropout=p_dropout),
                #SoftmaxLayer(n_in=fm3*img6*img6, n_out=10, p_dropout=p_dropout)], mini_batch_size)
                OutputLayer(n_in=fc, n_out=n_out, cost_type=cost_type, p_dropout=p_dropout)], mini_batch_size)
                    
            test_results = net.SGD(training_data, epochs, mini_batch_size, test_data, momentum, timeout=timeout, 
                                   lmbda=lmbda, add_noise=[], clip_weights=[], round_down=[])
            results.append(test_results)
            best_results_per_config.append(max(test_results))
            
    best_results.append(max(best_results_per_config))
    avg_results.append(sum(best_results_per_config)/num_sims)
    
    print "\nAverage: {:.2f}, Best: {:.2f}".format(sum(best_results_per_config)/num_sims, max(best_results_per_config))
    
    f = open(results_conv_0L_file, 'ab')
    f.write("{}: avg: {:.2f}, max: {:.2f}\r\n".format(dataset[-14:], avg_results[-1], best_results[-1]))
    f.close()
        
#print "\nbest_clipped_accuracy:", best_clipped_accuracy
#print "best_round_down_accuracy:", best_round_down_accuracy
#""" 
      
print '\nAvg: ',
for value in avg_results:
    print "{:.2f}".format(value),
print
print 'Max: ',
for value in best_results:
    print "{:.2f}".format(value),
print '\n\n\n'

#plot_total(epochs, results, momentums, 'momentum')
#plot_total(epochs, results, L2s, 'L2')
#plot_total(epochs, results, wscales, 'wscale')
#plot_total([95,100], epochs, results, fms, 'fm1')