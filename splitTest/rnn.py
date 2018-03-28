import numpy as np 
from fft import *
import operator
from datetime import datetime
import os

learning_rate = np.float32(0.001)
INPUT_DIM = 20000
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
        self.U = np.random.uniform(-np.sqrt(1./input_dim), np.sqrt(1./input_dim), (hidden_dim, input_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (output_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

    #Save parameters U, V, W
    def save_param(self, filename):
        np.savez(filename, U=self.U, V=self.V, W=self.W)

    #Load parameters back in
    def load_param(self, filename):
        npzfile = np.load(filename)
        self.U, self.V, self.W = npzfile["U"], npzfile["V"], npzfile["W"]
        # self.hidden_dim = self.U[0]
        # self.input_dim = self.U[1]
        # self.output_dim = self.V[0]

    #finding soft max
    def softmax(self, x):
        xt = np.exp(x - np.max(x)) #np.exp(x)
        return xt / np.sum(xt)

    def loss(self, x, y):
        probs = self.softmax(x)
        return -np.log(probs[y])

    #Time step/Input
    def forward_prop(self, x):
        T = len(x)
        if T == self.input_dim:
            x = [x]
            T = 1
        # s = hidden layer
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        # o = output layer
        o = np.zeros((T, self.output_dim))

        for t in np.arange(T):
            s[t] = np.tanh(np.matmul(self.U, x[t]) + self.W.dot(s[t-1])) #x[t] is floats rn
            #s[t] = np.tanh(self.U[:,t] * x[t] + self.W.dot(s[t-1]))
            o[t] = self.softmax(self.V.dot(s[t]))
        return [o, s]

    # argmax: axis = 0 means return index for vertical, axis = 1 means return index for horizontal
    # each hz gets an output, gets max for each and then returns the one that appears the most
    def predict(self, x):
        o, s = self.forward_prop(x)
        return np.argmax(o, axis=1)

    # Find total loss (loss helper)
    def calculate_total_loss(self, x, y):
        L = 0
        if len(x) == self.input_dim:
            x = [x]
        for i in np.arange(len(x)):
            o, s = self.forward_prop(x[i])
            for j in np.arange(len(y[i])):
                correct_word_predictions = o[j][np.flatnonzero(y[i][j])[0]]
                L += -1 * np.sum(np.log(correct_word_predictions))
        return L

    # Calculate loss
    def calculate_loss(self, x, y):
        N = np.sum((len(x_i) for x_i in x))
        #N = len(x)
        #print "N: %f" % N
        return self.calculate_total_loss(x,y) / N

    # Used to calculate the gradients
    # TODO: Convert from one entry = input to 10k = one input (WRONG?)
    def bptt(self, x, y):
        T = len(y)
        if len(x) == self.input_dim:
            T = 1
            x = [x]
            y = [y]
        # Set up gradients
        o, s = self.forward_prop(x)
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o - y
        # Looping through backwards
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T) 
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            # Back prop through time(at most bptt_truncate steps)
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                #print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                dLdW += np.outer(delta_t, s[bptt_step-1])
                dLdU += np.outer(delta_t, x[bptt_step]) # dLdU[:,x[bptt_step]] += delta_t
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
        return [dLdU, dLdV, dLdW]

    # One step of SGD
    def sgd_step(self, x, y, learning_rate):
        dU, dV, dW = self.bptt(x, y)
        self.U -= learning_rate * dU
        self.V -= learning_rate * dV
        self.W -= learning_rate * dW

    # Used to check if the calculated gradients are correct(VERY COMPUTIVE HEAVY)
    # Unsure if this is working or if bptt/forward prop is wrong...
    def gradient_checking(self, x, y, h=0.0001, error_threshold=0.01):
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
                    print "ERROR: parameter=%s idx=%s" % (pname, idx)
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

# Outer SGD Loop
# - model: The RNN model instance
# - X_train: The training data set
# - y_train: The training data labels
# - learning_rate: Initial learning rate for SGD
# - nepoch: Number of times to iterate through the complete dataset
# - evaluate_loss_after: Evaluate the loss after this many epochs
def train_with_sgd(model, x_train, y_train, learning_rate=0.01, nepoch=100, evaluate_loss_after=5):
    loss_history = []
    examples_seen = 0
    for epoch in range(nepoch):
        # Loss for plotting
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(x_train, y_train)
            loss_history.append((examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, examples_seen, epoch, loss)
            #model.save_param("rnn-theano-parameters.npz") 
            # Change learning if loss increase
            if (len(loss_history) > 1 and loss_history[-1][1] > loss_history[-2][1]):
                learning_rate = learning_rate * 0.5 
                print "Loss increased! Changed learning rate to %f" % learning_rate
            sys.stdout.flush()

        for i in range(len(y_train)):
            model.sgd_step(x_train[i], y_train[i], learning_rate)
            examples_seen += 1

def testing(model, x_test, y_test):
    num_correct = 0
    num_tested = 0
    for t in range(len(x_test)):
        prediction = model.predict([x_test[t]])
        if y_test[t][prediction[0]] == 1:
            num_correct += 1
        num_tested += 1
    acc = num_correct / num_tested * 100
    print "Num Correct: " + str(num_correct)
    print "Accuracy: %f" % acc



np.random.seed(10)
model = RNN()


dX = []
dY = []

for folder in os.listdir("data_one_octave/"):
    cor = 0
    test = 0
    notes = len(os.listdir("data_one_octave/" + folder))
    extra_X, extra_Y = get_data("data_one_octave/" + folder + "/data" + str(notes) + ".npz")
    shape_x, shape_Y = get_data("data_one_octave/" + folder + "/data1.npz")
    X_Song = np.zeros(shape=((notes - 1) * len(shape_x) + len(extra_X), INPUT_DIM))
    Y_Song = np.zeros(shape=((notes - 1) * len(shape_x) + len(extra_X), 12))
    print "Training on file " + str(folder)
    for i in range(len(os.listdir("data_one_octave/" + folder))):
        X, Y = get_data("data_one_octave/" + folder + "/data" + str(i + 1) + ".npz")
        X = X[...,:INPUT_DIM]
        for j in range(len(X)):
            X_Song[i * len(shape_x) + j] = X[j]
            Y_Song[i * len(shape_x) + j] = Y[j]
    print X_Song.shape
    print Y_Song.shape
    dX.append(np.float32(X_Song))
    dY.append(np.float32(Y_Song))

loss = model.calculate_loss(dX, dY)
print "Loss: %s" %(loss)


# for folder in os.listdir("data/"):
#     for filename in os.listdir("data/" + folder):
#         if filename.endswith(".npz"):
#             print "Training on file " + str(folder)
#             model.load_param("rnn-theano-parameters.npz")
#             X, Y = get_data("data/" + folder + "/" + filename)
#             print X.shape
#             print Y.shape
#             train_with_sgd(model, X, Y)


# X, Y = get_data("data1.npz")
# model.load_param("rnn-theano-parameters.npz")
# train_with_sgd(model, X, Y)
# testing(model, X, Y)

#To think about: Change Y from array to just one number for each note(check if chords are calcuated?)

# x, y = f('MAPS_ISOL_NO_P_S0_M61_AkPnBsdf_61')
# predictions = model.predict(x)
# print predictions

# o, s = model.forward_prop(X[8:10])
# np.savetxt('sampleS.txt', s, delimiter=',')
# model.save_param("./data/rnn-theano-data.npz")
# print o.shape
# print o

#Check calculate loss
# print "Expected Loss for random predictions: %f" % np.log(12)
# print "Actual loss: %f" % model.calculate_loss(X, Y)

# model.gradient_checking(X[0:3], Y[0:3])

# dU, dV, dW = model.bptt(X[0:5], Y[0:5])
# print dU
# print dV
# print dW