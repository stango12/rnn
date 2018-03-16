from pydub import AudioSegment
import os

# Takes MIDI data from txt and splits to notes
def splitData(directory, filename):
	song = AudioSegment.from_wav(directory + filename)
	info = directory + filename[:-4] + ".txt"
	with open(info, "r") as f:
		f.next()
		for line in f:
			numbers_float = map(float, line.split())
			minute = song[numbers_float[0] * 1000:numbers_float[1] * 1000]
			minute.export(directory + filename[:-4] + "/" + filename[:-4] + "_" + str(numbers_float[0] * 1000) + "_" + str(numbers_float[1] * 1000) + "_" + str(int(numbers_float[2])) + ".wav", format="wav")

#Create folders for each song
def multi_file(directory):
	for filename in os.listdir(directory):
	    if filename.endswith(".wav"):
	    	print "Working on " + filename[:-4]
	    	if not os.path.exists(directory + filename[:-4]):
	    		os.makedirs(directory + filename[:-4])
	    	splitData(directory, filename)
	    	os.rename(directory + filename, directory + filename[:-4] + "/" + filename)
	    	os.rename(directory + filename[:-4] + ".txt", directory + filename[:-4] + "/" + filename[:-4] + ".txt")
	    	os.remove(directory + filename[:-4] + ".mid")

#think this is just for the wav files that are only one note
def solo_notes(directory):
	for filename in os.listdir(directory):
		if filename.endswith(".wav"):
			print "Working on " + filename[:-4]
			os.rename(directory + filename, directory + filename[:-13] + ".wav")
			os.remove(directory + filename[:-4] + ".txt")
			os.remove(directory + filename[:-4] + ".mid")

multi_file("data/")
#solo_notes("data/ISO_Notes/")