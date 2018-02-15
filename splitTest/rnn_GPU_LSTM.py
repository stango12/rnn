import theano
import theano.tensor as T
import numpy as np 
import operator
from datetime import datetime
import time
import os

#TRY implementing ReLU instead of tanh and sigmoid
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda,floatX=float32"

learning_rate = np.float32(0.001)
class RNNTheano:
    #INPUT: 85,000 sized input array (0.1hz step size)
    #OUTPUT: 12 sized output array for each half note in an octave
    def __init__(self, input_dim=45000, hidden_dim=100, output_dim=12, bptt_truncate=4):
        #Assign instance variables
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.bptt_truncate = bptt_truncate
        #Randomly initialize parameters
        A = np.random.uniform(-np.sqrt(1./input_dim), np.sqrt(1./input_dim), (hidden_dim, input_dim))
        U = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (4, hidden_dim, hidden_dim)) #[i, f, o, g]
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (output_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (4, hidden_dim, hidden_dim))
        self.A = theano.shared(name='A', value=A.astype(theano.config.floatX))
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX)) 
        self.theano_setup() 

    def theano_setup(self):
        A, U, V, W = self.A, self.U, self.V, self.W
        x = T.fmatrix('x')
        y = T.fmatrix('y')

        #Using LSTM (Think about adding bias)
        def forward_prop_step(x_t, s_t_prev, c_t_prev, A, U, V, W):
            x_e = T.dot(A, x_t)
            i = T.nnet.hard_sigmoid(U[0].dot(x_e) + W[0].dot(s_t_prev))
            f = T.nnet.hard_sigmoid(U[1].dot(x_e) + W[1].dot(s_t_prev))
            o = T.nnet.hard_sigmoid(U[2].dot(x_e) + W[2].dot(s_t_prev))
            g = T.tanh(U[3].dot(x_e) + W[3].dot(s_t_prev))
            c_t = c_t_prev * f + g * i
            s_t = T.tanh(c_t) * o
            o_t = T.nnet.softmax(V.dot(s_t))
            return [o_t[0], s_t, c_t]

        [o, s, c], updates = theano.scan(forward_prop_step, sequences=x, outputs_info=[None, dict(initial=T.zeros(self.hidden_dim)), dict(initial=T.zeros(self.hidden_dim))], non_sequences=[A, U, V, W], truncate_gradient=self.bptt_truncate, strict=True)

        prediction = T.argmax(o, axis = 1)
        o_error = T.sum(T.nnet.categorical_crossentropy(o, y)) #need sum or nah?

        dA = T.grad(o_error, A)
        dU = T.grad(o_error, U)
        dV = T.grad(o_error, V)
        dW = T.grad(o_error, W)

        # theano functions (input, output)
        self.forward_prop = theano.function([x], o)
        self.bptt = theano.function([x, y], [dU, dV, dW, dA])
        self.predict = theano.function([x], prediction)
        self.error = theano.function([x, y], o_error)

        learning_rate = T.fscalar('learning_rate')
        self.sgd_step = theano.function([x, y, learning_rate], [], 
                                    updates=[(self.U, self.U - learning_rate * dU),
                                             (self.V, self.V - learning_rate * dV),
                                             (self.W, self.W - learning_rate * dW), 
                                             (self.A, self.A - learning_rate * dA)])

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

    # Calculate loss
    def calculate_loss(self, x, y):
        #N = np.sum((len(x_i) for x_i in x))
        N = len(x)
        #print "N: %f" % N
        return np.sum(self.error([x[i]], [y[i]]) for i in range(N)) / N

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
def train_with_sgd(model, x_train, y_train, learning_rate=np.float32(0.01), nepoch=100, evaluate_loss_after=5):
    loss_history = []
    examples_seen = 0
    for epoch in range(nepoch):
        # Loss for plotting
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(x_train, y_train)
            loss_history.append((examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, examples_seen, epoch, loss)
            #model.save_param("rnn-theano-parameters45k-LSTM.npz") 
            # Change learning if loss increase
            if (len(loss_history) > 1 and loss_history[-1][1] > loss_history[-2][1]):
                learning_rate = learning_rate * 0.5 
                print "Loss increased! Changed learning rate to %f" % learning_rate
            # sys.stdout.flush()

        for i in range(len(y_train)):
            model.sgd_step([x_train[i]], [y_train[i]], learning_rate)
            examples_seen += 1

np.random.seed(10)
model = RNNTheano()

for folder in os.listdir("data/"):
    notes = len(os.listdir("data/" + folder))
    extra_X, extra_Y = get_data("data/" + folder + "/data" + str(notes) + ".npz")
    shape_x, shape_Y = get_data("data/" + folder + "/data1.npz")
    X_Song = np.zeros(shape=((notes - 1) * len(shape_x) + len(extra_X) + 2, 45000))
    Y_Song = np.zeros(shape=((notes - 1) * len(shape_x) + len(extra_X) + 2, 12))
    for i in range(len(os.listdir("data/" + folder))):
        print "Training on file " + str(folder)
        #model.load_param("rnn-theano-parameters45k-LSTM.npz")
        X, Y = get_data("data/" + folder + "/data" + str(i + 1) + ".npz")
        X = X[...,:45000]
        for j in range(len(X)):
            X_Song[i * len(shape_x) + 1 + j] = X[j]
            Y_Song[i * len(shape_x) + 1 + j] = Y[j]
    print X_Song.shape
    print Y_Song.shape
    train_with_sgd(model, np.float32(X_Song), np.float32(Y_Song), learning_rate)


# X, Y = get_data("data/MAPS_MUS-ty_mai_AkPnBcht/data1.npz")
# X = X[...,:45000]
# X1, Y1 = get_data("data/MAPS_MUS-ty_mai_AkPnBcht/data2.npz")
# X1 = X1[...,:45000]
# X2, Y2 = get_data("data/MAPS_MUS-ty_mai_AkPnBcht/data3.npz")
# X2 = X2[...,:45000]

# notes = len(X) + len(X1) + len(X2) + 2
# X_Song = np.zeros(shape=(notes, 45000))
# Y_Song = np.zeros(shape=(notes, 12))

# for i in range(len(X)):
#     X_Song[i + 1] = X[i]
#     Y_Song[i + 1] = Y[i]

# for i in range(len(X1)):
#     X_Song[len(X) + 1] = X1[i]
#     Y_Song[len(Y) + 1] = Y1[i]

# for i in range(len(X2)):
#     X_Song[len(X) + len(X1) + 1] = X2[i]
#     Y_Song[len(Y) + len(Y1) + 1] = Y2[i]

# print X_Song.shape
# print Y_Song.shape
# train_with_sgd(model, np.float32(X_Song), np.float32(Y_Song))


# t1 = time.time()
# model.sgd_step([X[10]], [Y[10]], learning_rate)
# t2 = time.time()
# print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)

#model.save_param("/data/rnn-theano-parameters.npz")
