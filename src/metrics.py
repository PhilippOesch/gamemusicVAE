import numpy as np

num_notes = 88

def tonal_distance(samples, thresh=0.5):

    # tranform to chroma vector of size 12
    eq_t_samples= np.zeros((samples.shape[0], samples.shape[1], 12), dtype= np.float32)   # 12 Tone Equal Temperament
    print(samples.shape)
    for sample in samples:
        for y in range(sample.shape[0]):
            note= np.zeros((12), dtype= np.float32)
            for x in range(sample.shape[1]):
                i= (x-2) % 12   # -2 Piano starts at note-A instead of note-C
                # print(sample[y, x].dtype)
                if sample[y, x]> note[i]:
                    note[i]= sample[y, x]
            eq_t_samples[sample, y]= note

    print(eq_t_samples)

def get_transformation_matrix(l):
    r1= 1
    r2= 1
    r3= 0.5

    return np.array(
        (r1* np.sin(l* (7*np.pi/ 6)),
        r1* np.cos(l* ()),
        ))

