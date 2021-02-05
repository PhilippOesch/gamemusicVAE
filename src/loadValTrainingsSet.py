import utility.midi as midi
import os
import utility.util as util
import numpy as np

patterns = {}
directory = "../sampleMidis/battlethemes"
all_samples = []
song_num= 0
print ("Loading Songs...")

for root, subdirs, files in os.walk(directory):
    for file in files:
        path = root+ "\\" + file
        if not (path.endswith('.mid') or path.endswith('.midi')):
            continue

        try:
            samples = midi.midi_to_samples(path)[:16] # only get the first 16 samples of each file so that in the evaluation 2 set of the same song cant be compared to each other 

        # samples = midi.midi_to_samples(path)
        except:
            print ("ERROR", path)
            continue

        if len(samples)< 16:
            continue

        if samples == []:
            continue

        samples= util.centered_transposed(samples)
        all_samples+= samples
        song_num+= 1;

all_samples = np.reshape(all_samples, (song_num, 16, 96, 88))
print(all_samples.shape)
np.save('../evaluation_sets/battle_theme/train_set_samples', all_samples)
print ("Done")