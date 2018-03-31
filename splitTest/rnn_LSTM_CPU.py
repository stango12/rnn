import numpy as np 
import operator
from fft import *
from datetime import datetime
import time
import os
import sys

INPUT_DIM = 20000
learning_rate = np.float32(0.001)
load_save = True

class RNN:
    #INPUT: 85,000 sized input array (0.1hz step size)
    #OUTPUT: 12 sized output array for each half note in an octave
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=100, output_dim=12, bptt_truncate=4):
        #Assign instance variables
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.bptt_truncate = bptt_truncate
        #Randomly initialize parameters
        self.A = np.random.uniform(-np.sqrt(1./input_dim), np.sqrt(1./input_dim), (hidden_dim, input_dim))
        self.U = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (4, hidden_dim, hidden_dim)) #[i, f, o, g]
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (output_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (4, hidden_dim, hidden_dim))
        #biases
        self.b = np.zeros((4, hidden_dim))
        self.c_o = np.zeros(output_dim)

    def forward_prop(self, x):
        A, U, V, W, b, c_o = self.A, self.U, self.V, self.W, self.b, self.c_o
        T = len(x)
        s = np.zeros((T, self.hidden_dim))
        out = np.zeros((T, self.output_dim))
        c = np.zeros((T, self.hidden_dim))

        for t in np.arange(T):
            x_e = A.dot(x[t])
            i = self.hard_sigmoid(U[0].dot(x_e) + W[0].dot(s[t-1]) + b[0]) #T.nnet.hard_sigmoid
            f = self.hard_sigmoid(U[1].dot(x_e) + W[1].dot(s[t-1]) + b[1]) #T.nnet.hard_sigmoid
            o = self.hard_sigmoid(U[2].dot(x_e) + W[2].dot(s[t-1]) + b[2]) #T.nnet.hard_sigmoid
            g = np.tanh(U[3].dot(x_e) + W[3].dot(s[t-1]) + b[2])
            c[t] = c[t-1] * f + g * i
            s[t] = np.tanh(c[t]) * o # was tanh, changed to reLU due to vanishing gradient problem T.nnet.relu(c_t) * o
            out[t] = self.softmax(V.dot(s[t]) + c_o)
        return [out, s]

    def hard_sigmoid(self, x):
        slope = 0.2
        shift = 0.5
        x = (x * slope) + shift
        x = np.clip(x, 0, 1)
        return x

    #finding soft max
    def softmax(self, x):
        xt = np.exp(x - np.max(x)) #np.exp(x)
        return xt / np.sum(xt)

    def predict(self, x):
        o, s = self.forward_prop(x)
        return np.argmax(o, axis=1)

    #Save parameters U, V, W
    def save_param(self, filename):
        A = self.A.get_value()
        U = self.U.get_value()
        V = self.V.get_value()
        W = self.W.get_value()
        b = self.b.get_value()
        c_o = self.c_o.get_value()
        np.savez(filename, A=A, U=U, V=V, W=W, b=b, c_o=c_o)

    #Load parameters back in
    def load_param(self, filename):
        npzfile = np.load(filename)
        self.A, self.U, self.V, self.W, self.b, self.c_o = npzfile["A"], npzfile["U"], npzfile["V"], npzfile["W"], npzfile["b"], npzfile["c_o"]


#Getting from all data
def get_data(filename):
    npzfile = np.load(filename)
    X, Y = npzfile["data"], npzfile["out"]
    return X, Y

np.random.seed(10)
model = RNN()

if load_save:
    model.load_param("rnn-theano-parameters-one-octave-songs-2.npz")
#     #model.load_param("rnn-theano-parameters45k-LSTM.npz")

#single_note_test(model, "note_test")

X, Y = get_data("dirty_example.npz")
o, s = model.forward_prop(np.float32(X))
print o
print model.predict(np.float32(X))
