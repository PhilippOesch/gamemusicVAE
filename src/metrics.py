import warnings
import numpy as np

num_notes = 88

warnings.filterwarnings("ignore",category =RuntimeWarning)

def tonal_distance(samples, ignore_treshhold= True, thresh=0.5):

    # tranform to chroma vector of size 12
    # 12 Tone Equal Temperament
    eq_t_samples = np.zeros(
        (samples.shape[0], samples.shape[1], 12), dtype=np.float32)

    # 6d Vectors for tonal centroid calculation
    t_c_samples = np.zeros(
        (samples.shape[0], samples.shape[1], 6), dtype=np.float32)

    # print(samples.shape)
    for z in range(samples.shape[0]):
        for y in range(samples.shape[1]):
            note = np.zeros((12), dtype=np.float32)
            for x in range(samples.shape[2]):
                i = (x-2) % 12   # -2 Piano starts at note-A instead of note-C
                # print(sample[y, x].dtype)
                if ignore_treshhold and samples[z, y, x] > note[i]:
                    note[i] = samples[z, y, x]
                elif not ignore_treshhold and samples[z, y, x]> thresh:
                    note[i]= 1
            eq_t_samples[z, y] = note

    # print(eq_t_samples)

    for z in range(eq_t_samples.shape[0]):
        for y in range(eq_t_samples.shape[1]):
            y_d6= [get_transformation_matrix(x)* eq_t_samples[z ,y, x] for x in range(eq_t_samples.shape[2])]
            sum_y= np.sum(y_d6, axis= 0);
            # print(sum_y)
            l_norm= np.linalg.norm(eq_t_samples[z, y])
            zeta_y= sum_y * (1/ l_norm)
            zeta_y= np.nan_to_num(zeta_y)
            t_c_samples[z, y] = zeta_y

    print(t_c_samples)


def get_transformation_matrix(l):
    r1 = 1
    r2 = 1
    r3 = 0.5

    t_matrix= np.array(
        (r1 * np.sin(l * (7*np.pi / 6)),
         r1 * np.cos(l * (7*np.pi / 6)),
         r2 * np.sin(l * (3*np.pi / 2)),
         r2 * np.cos(l * (3*np.pi / 2)),
         r3 * np.sin(l * (2*np.pi / 3)),
         r3 * np.cos(l * (2*np.pi / 3))
         ))

    return t_matrix
