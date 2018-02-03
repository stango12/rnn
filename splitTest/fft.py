import matplotlib.pyplot as plt
import numpy as np
import scipy
import os
from scipy import signal
from scipy.io import wavfile # get the api
from scipy.fftpack import fft
from pydub import AudioSegment
from pylab import *

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
    test = 10000 * len(c) / fs #limits the size of window to 5000hz
    # plt.plot(abs(c[:(test)]),'r')  #was d-1, gets the frequency bins to be from 0-5000hz

    # plt.plot(np.log10(abs(c[:(test)]) * 1 + 1),'r')
    # plt.xlabel('Frequency Bin')
    # plt.ylabel('Magnitude')
    # plt.show()

    # savefig(filename+'.png',bbox_inches='tight')
    if filename.endswith(".wav"):
        return np.log10(abs(c[:(test)]) * 1 + 1), set_output(filename[-6:-4])
    else:
        return np.log10(abs(c[:(test)]) * 1 + 1), set_output(filename[-2:])

def f2(filename):
    fs, data = wavfile.read(filename +  ".wav") # load the data
    fft_out = fft(data)
    freqs = scipy.fftpack.fftfreq(len(fft_out))
    idx = np.argmax(np.abs(fft_out))
    freq = freqs[idx]
    freq_in_hz = abs(freq * fs)
    print(freq_in_hz)

def preprocess(directory):
    x = []
    y = []
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            print "Working on " + filename
            data, out = f(filename, directory)
            x.append(data)
            y.append(out)
        else:
            continue
    np.savez("data.npz", data = x, out = y)
    return x, y

if __name__ == '__main__':
    x, y = f('MAPS_ISOL_NO_P_S0_M61_AkPnBsdf_61')
    print y
	#preprocess('data/bach_846')