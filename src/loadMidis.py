import midi
import os
import util
import numpy as np

patterns = {}
directory = "../sampleMidis/overworldTheme"
all_samples = []
all_lens = []
print ("Loading Songs...")

if not os.path.exists('../model'):
    os.makedirs('../model')

for root, subdirs, files in os.walk(directory):
    for file in files:
        path = root+ "\\" + file
        if not (path.endswith('.mid') or path.endswith('.midi')):
            continue

        try:
            samples = midi.midi_to_samples(path)

        # samples = midi.midi_to_samples(path)
        except:
            print ("ERROR", path)
            continue

        if len(samples)< 8:
            continue

        if samples == []:
            continue
        samples, lens = util.generate_add_centered_transpose(samples)
        # samples, lens= util.generate_all_transpose(samples, radius=6)
        all_samples += samples
        all_lens += lens 
        print (len(all_lens))
	
assert(sum(all_lens) == len(all_samples))
print ("Saving " + str(len(all_samples)) + " samples...")
all_samples = np.array(all_samples, dtype=np.uint8)
all_lens = np.array(all_lens, dtype=np.uint32)
# np.save('../model/samples.npy', all_samples)
# np.save('../model/lengths.npy', all_lens)
np.save('../model/samples_untransposed.npy', all_samples)
np.save('../model/lengths_untransposed.npy', all_lens)
print ("Done")