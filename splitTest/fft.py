# import matplotlib.pyplot as plt
import numpy as np
import scipy
import os
import os.path
from scipy import signal
from scipy.io import wavfile # get the api
from scipy.fftpack import fft
from pydub import AudioSegment
from pylab import *

notes_data = []

#setting up output array, size 12 for each halfstep in octave, output start at C then to B
def set_output(note_num):
    nn = int(note_num)
    o = np.zeros(12)
    o[nn % 12] = 1
    return o

#loads the signal and sets the fft size to 441000 so bins are 0.1hz
def f(filename, directory=None):
    if filename.endswith(".wav"):
        if directory is None:
            fs, data = wavfile.read(filename)
        else:
            fs, data = wavfile.read(directory + "/" + filename)
    else:
        if directory is None:
            fs, data = wavfile.read(filename + ".wav") # load the data
        else:
            fs, data = wavfile.read(directory + "/" + filename + ".wav")
    
    a = data.T[0] # this is a two channel soundtrack, I get the first track
    hamming = scipy.signal.hamming(len(a))
    b=[(ele/2**8.)*2-1 for ele in a] # this is 8-bit track, b is now normalized on [-1,1)
    b  = b * hamming
    c = fft(b, 441000) # create a list of complex number
    d = len(c)/2  # you only need half of the fft list
    # c = np.abs(c)
    # minV = np.amin(c)
    # maxV = np.amax(c)
    # plt.plot(abs(c[:(d-1)]),'r') 
    # print maxV
    # for i in range(len(c)):
    # 	if c[i] == maxV:
    # 		print i
    # plt.plot((c[:(d-1)] - minV) / (maxV - minV),'r')

    # print str(len(data))
    # print str(len(c))
    # print str(fs)
    # print str(float(fs) / len(c)) #scale for each frequency "bin"
    # for i in range(d): #prints the hz of ones that are above a certain range
    #     if c[i] > 1000:
    #         print(str(i * float(fs) / len(c)) + "Hz")

    freqs = scipy.fftpack.fftfreq(d - 1, 2 / float(fs))
    # plt.xticks(np.arange(0, len(data) / 100 + 1, float(fs) / len(data)))
    test = 2000 * len(c) / fs #limits the size of window to 5000hz
    plt.plot(abs(c[:(test)]),'r')  #was d-1, gets the frequency bins to be from 0-5000hz

    plt.plot(np.log10(abs(c[:(test)]) * 1 + 1),'r')
    plt.xlabel('Frequency Bin')
    plt.ylabel('Magnitude')
    plt.show()

    # savefig(filename+'.png',bbox_inches='tight')
    if filename.endswith(".wav"):
        return np.log10(abs(c[:(test)]) * 1 + 1), set_output(filename[-15:-13]), int(filename[-15:-13]) % 12
    else:
        return np.log10(abs(c[:(test)]) * 1 + 1), set_output(filename[-2:]), 0 #-11:-9

def f2(filename):
    fs, data = wavfile.read(filename +  ".wav") # load the data
    fft_out = fft(data)
    freqs = scipy.fftpack.fftfreq(len(fft_out))
    idx = np.argmax(np.abs(fft_out))
    freq = freqs[idx]
    freq_in_hz = abs(freq * fs)
    print(freq_in_hz)

def load_data(option):
    if option == "clean":
        for i in range(12):
            npzfile = np.load("Solo Notes/note" + str(i) + ".npz")
            X = npzfile["data"]
            notes_data.append(X)

def preprocess(directory):
    x = []
    y = []
    count = 1
    for folder in os.listdir(directory): 
        x = []
        y = []
        count = 1
        if not os.path.isfile(directory + folder + "/data1.npz"):
            for filename in os.listdir(directory + folder):
                if filename.endswith(".wav") and filename[-6:-4].isdigit():
                    print "Working on " + filename
                    data, out = f(filename, directory + folder)
                    # for t in range(len(data)):
                    #     if data[t] > 2.5:
                    #         data[t] = 1
                    #     else:
                    #         data[t] = 0
                    x.append(data)
                    y.append(out)
                    if len(x) == 500:
                        print "Data Out"
                        np.savez(directory + folder + "/data" + str(count) + ".npz", data = x, out = y)
                        count += 1
                        x = []
                        y = []
                else:
                    continue
            print "Files: " + str((count - 1) * 500 + len(x))
            np.savez(directory + folder + "/data" + str(count) + ".npz", data = x, out = y)
    return x, y

def one_octave_preprocess(directory):
    x = []
    y = []
    count = 1
    load_data("clean")
    for folder in os.listdir(directory):
        x = []
        y = []
        count = 1
        for filename in os.listdir(directory + folder):
            if filename.endswith(".wav") and filename[-6:-4].isdigit():
                print "Working on " + filename
                nn = int(filename[-6:-4])
                o = np.zeros(12)
                o[nn % 12] = 1
                x.append(notes_data[nn % 12])
                y.append(o)
                if len(x) == 1000:
                    print "Data Out"
                    np.savez(directory + folder + "/data" + str(count) + ".npz", data = x, out = y)
                    count += 1
                    x = []
                    y = []
            else:
                continue
        print "Files: " + str((count - 1) * 500 + len(x))
        np.savez(directory + folder + "/data" + str(count) + ".npz", data = x, out = y)

if __name__ == '__main__':
    x, y, nn = f('MAPS_MUS-bach_847_AkPnBcht_1886.33_2022.11_60')
    np.savez("dirty_example.npz", data=x, out=y)

    # print x.shape
    # print y.shape
	# preprocess('data/')

    # for filename in os.listdir('Solo Notes/'):
    #     if filename.endswith(".wav"):
    #         x, y, nn = f(filename, "Solo Notes/")
    #         np.savez("Solo Notes/note" + str(nn) + ".npz", data=x)

    #one_octave_preprocess("New Folder/")
