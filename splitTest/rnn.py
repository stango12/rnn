import numpy as np 
from fft import *

learning_rate = 0.01
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

	#Save parameters U, V, W
	def save_param(self, filename):
		np.savez(filename, U=self.U, V=self.V, W=self.W)
		print self.U

	#Load parameters back in
	def load_param(self, filename):
		npzfile = np.load(filename)
		self.U, self.V, self.W = npzfile["U"], npzfile["V"], npzfile["W"]
		self.hidden_dim = self.U[0]
		self.input_dim = self.U[1]
		self.output_dim = self.V[0]

	#finding soft max
	def softmax(self, x):
		xt = np.exp(x) #np.exp(x - np.max(x))
		return xt / np.sum(xt)

	def loss(self, x, y):
		probs = self.softmax(x)
		return -np.log(probs[y])

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
	# each hz gets an output, gets max for each and then returns the one that appears the most
	def predict(self, x):
		o, s = self.forward_prop(x)
		a = np.argmax(o, axis=1)
		counts = np.bincount(a)
		return np.argmax(counts)

	# Find total loss (loss helper)
	def calculate_total_loss(self, x, y):
		L = 0
		for i in np.arange(len(y)):
			o, s = self.forward_prop(x[i])
			correct_word_predictions = o[:, np.flatnonzero(y[i])[0]]
			print np.flatnonzero(y[i])[0]
			L += -1 * np.sum(np.log(correct_word_predictions))
		return L

	# Calculate loss
	def calculate_loss(self, x, y):
		N = np.sum((len(x_i) for x_i in x))
		print "N: %f" % N
		return self.calculate_total_loss(x,y) / N

	# Used to calculate the gradients
	def bptt(self, x, y):
		T = len(y)
		o, s = self.forward_prop(x)
		# Set up gradients
		dLdU = np.zeros(self.U.shape)
		dLdV = np.zeros(self.V.shape)
		dLdW = np.zeros(self.W.shape)
		delta_o = o
		for i in np.arange(len(y)):
			delta_o[:, np.flatnonzero(y[i])[0]] -= 1.
		# Looping through backwards
		for t in np.arange(T)[::-1]:
			dLdV += np.outer(delta_o[t], s[t].T) 
			delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
			# Back prop through time(at most bptt_truncate steps)
			for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
	            print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
	            dLdW += np.outer(delta_t, s[bptt_step-1])              
	            dLdU[:,x[bptt_step]] += delta_t
	            # Update delta for next step
	            delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
	    return [dLdU, dLdV, dLdW]

	# Used to check if the calculated gradients are correct(VERY COMPUTIVE HEAVY)
	def gradient_checking(self, x, y, h=0.001, error_threshold=0.01):
		bptt_gradients = self.bptt(x, y)
		model_param = ['U', 'V', 'W']
		# Have to loop ahd check every param
		for pidx, pname in enumerate(model_param):
			parameter = operator.attrgetter(pname)(self)
        	print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
        	# Iterate through each element
        	it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
        	while not it.finished:
        		idx = it.multi_index
        		og_value = parameter[idx]
        		# Use f(x + h) - f(x - h) / (2 * h) to estimate gradient
        		parameter[idx] = og_value + h
        		gradplus = self.calculate_total_loss([x],[y])
        		parameter[idx] = og_value - h
        		gradminus = self.calculate_total_loss([x], [y])
        		estimated_grad = (gradplus - gradminus)/(2 * h)
        		parameter[idx] = og_value
        		backprop_grad = bptt_gradients[pidx][idx]
        		# Relative error = (|x - y|/(|x| + |y|))
        		relative_error = np.abs(backprop_grad - estimated_grad) / (np.abs(backprop_grad) + np.abs(estimated_grad))
        		if relative_error > error_threshold:
	                print "RROR: parameter=%s ix=%s" % (pname, ix)
	                print "+h Loss: %f" % gradplus
	                print "-h Loss: %f" % gradminus
	                print "Estimated_gradient: %f" % estimated_grad
	                print "Backpropagation gradient: %f" % backprop_grad
	                print "Relative Error: %f" % relative_error
	                return	
	            it.iternext()
	        print "Gradient check passed for %s" % (pname)	
	        
#Getting from all data
def get_data(filename):
	npzfile = np.load(filename)
	X, Y = npzfile["data"], npzfile["out"]
	return X, Y

np.random.seed(10)
model = RNN()

X, Y = get_data("data.npz")
print X.shape
print Y.shape
print Y[0]
#To think about: Change Y from array to just one number for each note(check if chords are calcuated?)

# Limit to 1000 examples to save time
print "Expected Loss for random predictions: %f" % np.log(12)
print "Actual loss: %f" % model.calculate_loss(X, Y)

# x, y = f('MAPS_ISOL_NO_P_S0_M61_AkPnBsdf_61')
# predictions = model.predict(x)
# print predictions

# X, Y = get_data("data.npz")
# print X.shape
# print Y.shape
# o, s = model.forward_prop(X[10])
# model.save_param("./data/rnn-theano-data.npz")
# print o.shape
# print o