import numpy as np 
from fft import *

class RNN:
	#INPUT: 100,000 sized input array (0.01hz step size)
	#OUTPUT: 12 sized output array for each half note in an octave
	def __init__(self, input_dim=100000, hidden_dim=100, output_dim=12, bptt_truncate=4):
		#Assign instance variables
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.bptt_truncate = bptt_truncate
		#Randomly initialize parameters
		self.U = np.random.uniform(-np.sqrt(1./input_dim), np.sqrt(1./input_dim), (hidden_dim, input_dim))
		self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (output_dim, hidden_dim))
		self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

	#finding soft max
	def softmax(self, x):
		xt = np.exp(x) #np.exp(x - np.max(x))
		return xt / np.sum(xt)

	#Time step/Input
	def forward_prop(self, x):
		T = len(x)
		# s = hidden layer
		s = np.zeros((T + 1, self.hidden_dim))
		s[-1] = np.zeros(self.hidden_dim)
		# o = output layer
		o = np.zeros((T, self.output_dim))

		for t in np.arange(T):
			s[t] = np.tanh(self.U[:,t] * x[t] + self.W.dot(s[t-1])) #x[t] is floats rn
			o[t] = self.softmax(self.V.dot(s[t]))
		return [o, s]

	# argmax: axis = 0 means return index for vertical, axis = 1 means return index for horizontal
	def predict(self, x):
		o, s = self.forward_prop(x)
		return np.argmax(o, axis=1)

np.random.seed(10)
model = RNN()
X = f('bach_846_602_945_60')
o, s = model.forward_prop(X)
print o.shape
print o