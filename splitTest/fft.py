import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import signal
from scipy.io import wavfile # get the api
from scipy.fftpack import fft
from pydub import AudioSegment
from pylab import *

#loads the signal and sets the fft size to 441000 so bins are 0.1hz
def f(filename):
    fs, data = wavfile.read(filename +  ".wav") # load the data
    print "Sampling Freq: " + str(fs)
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
    #plt.plot((c[:(d-1)] - minV) / (maxV - minV),'r')
    print str(len(data))
    print str(len(c))
    print str(fs)
    print str(float(fs) / len(c)) #scale for each frequency "bin"
    for i in range(d): #prints the hz of ones that are above a certain range
        if c[i] > 1000:
            print(str(i * float(fs) / len(c)) + "Hz")
    freqs = scipy.fftpack.fftfreq(d - 1, 2 / float(fs))
    #plt.xticks(np.arange(0, len(data) / 100 + 1, float(fs) / len(data)))
    test = 10000 * len(c) / fs #limits the size of window to 5000hz
    #plt.plot(abs(c[:(test)]),'r')  #was d-1, gets the frequency bins to be from 0-5000hz
    plt.plot(np.log10(abs(c[:(test)]) * 1 + 1),'r')
    plt.xlabel('Frequency Bin')
    plt.ylabel('Magnitude')
    plt.show()
    #savefig(filename+'.png',bbox_inches='tight')

def f2(filename):
    fs, data = wavfile.read(filename +  ".wav") # load the data
    fft_out = fft(data)
    freqs = scipy.fftpack.fftfreq(len(fft_out))
    idx = np.argmax(np.abs(fft_out))
    freq = freqs[idx]
    freq_in_hz = abs(freq * fs)
    print(freq_in_hz)


def split(filename):
	song = AudioSegment.from_wav(filename + ".wav")
	minute = song[1287:1630]
	minute.export(filename + "3.wav", format="wav")
	rate = 200
	one = 0
	two = rate
	i = 0
	# while two < 10 * 200:
	# 	minute = song[one:two]
	# 	minute.export(filename + str(i) + ".wav", format="wav")
	# 	i += 1
	# 	one = two
	# 	two += rate
	# lastMin = song[one:]
	# lastMin.export(filename + "_end.wav", format="wav")

if __name__ == '__main__':
	f('bach_846_602_945_60')