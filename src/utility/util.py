
import numpy as np
import cv2
import os


def transpose_range(samples):
    merged_sample = np.zeros_like(samples[0])
    for sample in samples:
        merged_sample = np.maximum(merged_sample, sample)
    merged_sample = np.amax(merged_sample, axis=0)
    min_note = np.argmax(merged_sample)
    max_note = merged_sample.shape[0] - np.argmax(merged_sample[::-1])
    return min_note, max_note


def generate_add_centered_transpose(samples):
    num_notes = samples[0].shape[1]
    min_note, max_note = transpose_range(samples)
    s = num_notes/2 - (max_note + min_note)/2
    out_samples = samples
    out_lens = [len(samples), len(samples)]
    for i in range(len(samples)):
        out_sample = np.zeros_like(samples[i])
        out_sample[:, int(min_note+s):int(max_note+s)
                   ] = samples[i][:, min_note:max_note]
        out_samples.append(out_sample)
    return out_samples, out_lens


def generate_all_transpose(samples, radius=6):
    num_notes = samples[0].shape[1]
    min_note, max_note = transpose_range(samples)
    min_shift = -min(radius, min_note)
    max_shift = min(radius, num_notes - max_note)
    out_samples = []
    out_lens = []
    for s in range(min_shift, max_shift):
        for i in range(len(samples)):
            out_sample = np.zeros_like(samples[i])
            out_sample[:, min_note+s:max_note +
                       s] = samples[i][:, min_note:max_note]
            out_samples.append(out_sample)
        out_lens.append(len(samples))
    return out_samples, out_lens


def centered_transposed(samples):
    num_notes = samples[0].shape[1]
    min_note, max_note = transpose_range(samples)
    s = num_notes/2 - (max_note + min_note)/2
    out_samples = []
    for i in range(len(samples)):
        out_sample = np.zeros_like(samples[i])
        out_sample[:, int(min_note+s):int(max_note+s)
                   ] = samples[i][:, min_note:max_note]
        out_samples.append(out_sample)
    return out_samples

def sample_to_pic(fname, sample, thresh=None):
    if thresh is not None:
        inverted = np.where(sample > thresh, 0, 1)
    else:
        inverted = 1.0 - sample
    cv2.imwrite(fname, inverted * 255)


def samples_to_pics(dir, samples, thresh=None):
    if not os.path.exists(dir):
        os.makedirs(dir)
    for i in range(samples.shape[0]):
        sample_to_pic(dir + '/s' + str(i) + '.png', samples[i], thresh)


