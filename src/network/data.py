import numpy as np


class Data:
    def __init__(self, datasetName="battle", max_length=16):
        self.num_offsets = 1

        self.y_samples = np.load(
            '../model/datasets/' + datasetName + '_samples.npy')
        self.y_lengths = np.load(
            '../model/datasets/' + datasetName + '_lengths.npy')
        self.num_samples = self.y_samples.shape[0]
        self.num_songs = self.y_lengths.shape[0]
        assert(np.sum(self.y_lengths) == self.num_samples)

        self.y_shape = (self.num_songs * self.num_offsets,
                        max_length) + self.y_samples.shape[1:]
        self.y_orig = np.zeros(self.y_shape, dtype=np.float32)
        self.cur_ix = 0

        for i in range(self.num_songs):
            for ofs in range(self.num_offsets):
                ix = i * self.num_offsets + ofs
                end_ix = self.cur_ix + self.y_lengths[i]
                for j in range(max_length):
                    k = (j + ofs) % (end_ix - self.cur_ix)
                    self.y_orig[ix, j] = self.y_samples[self.cur_ix + k]
            self.cur_ix = end_ix
        assert(end_ix == self.num_samples)

        self.y_train = np.copy(self.y_orig)

    def get_train_set(self):
        return self.y_train, self.y_shape
