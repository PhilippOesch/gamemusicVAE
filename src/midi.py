from mido import MidiFile, MidiTrack, Message
import numpy as np
import string

num_notes = 88
samples_per_measure = 96


def msg2dict(msg):
    result = dict()
    if 'note_on' in msg:
        on_ = True
    elif 'note_off' in msg:
        on_ = False
    else:
        on_ = None
    result['time'] = int(msg[msg.rfind('time'):].split(' ')[0].split('=')[1].translate(
        str.maketrans({a: None for a in string.punctuation})))

    if on_ is not None:
        for k in ['note', 'velocity']:
            result[k] = int(msg[msg.rfind(k):].split(' ')[0].split('=')[1].translate(
                str.maketrans({a: None for a in string.punctuation})))
    return [result, on_]


def switch_note(last_state, note, velocity, on_=True):
    # piano has 88 notes, corresponding to note id 21 to 108, any note out of this range will be ignored
    result = [0] * num_notes if last_state is None else last_state.copy()
    if 21 <= note <= 108:
        result[note-17] = velocity if on_ else 0
    return result


def get_new_state(new_msg, last_state):
    new_msg, on_ = msg2dict(str(new_msg))
    new_state = switch_note(
        last_state, note=new_msg['note'], velocity=new_msg['velocity'], on_=on_) if on_ is not None else last_state
    return [new_state, new_msg['time']]


def track2seq(track, ticks_per_beat, ticks_per_measure):
    result = []
    last_state, last_time = get_new_state(str(track[0]), [0]*num_notes)
    for i in range(1, len(track)):
        new_state, new_time = get_new_state(track[i], last_state)
        abs_time = new_time * samples_per_measure / ticks_per_measure
        if new_time > 0:
            result += [last_state] * round(abs_time)
        last_state, last_time = new_state, abs_time
    return result


def midi_to_samples(fname, min_msg_pct=0.1):
    # has_time_sig = False
    # flag_warning = False
    # mid = MidiFile(fname, clip=True)
    # ticks_per_beat = mid.ticks_per_beat
    # ticks_per_measure = 4 * ticks_per_beat

    # for i, track in enumerate(mid.tracks):
    #     for msg in track:
    #         if msg.type == 'time_signature':
    #             new_tpm = msg.numerator * ticks_per_beat * 4 / msg.denominator
    #             if has_time_sig and new_tpm != ticks_per_measure:
    #                 flag_warning = True
    #             ticks_per_measure = new_tpm
    #             has_time_sig = True
    # if flag_warning:
    #     print("  ^^^^^^ WARNING ^^^^^^")
    #     print("    " + fname)
    #     print("    Detected multiple distinct time signatures.")
    #     print("  ^^^^^^ WARNING ^^^^^^")
    #     return []

    # tracks_len = [len(tr) for tr in mid.tracks]
    # min_n_msg = max(tracks_len) * min_msg_pct
    # # convert each track to nested list
    # all_arys = []
    # for i in range(len(mid.tracks)):
    #     if len(mid.tracks[i]) > min_n_msg:
    #         ary_i = track2seq(mid.tracks[i], ticks_per_beat, ticks_per_measure)
    #         all_arys.append(ary_i)
    # # make all nested list the same length
    # max_len = max([len(ary) for ary in all_arys])
    # for i in range(len(all_arys)):
    #     if len(all_arys[i]) < max_len:
    #         all_arys[i] += [[0] * num_notes] * (max_len - len(all_arys[i]))
    # all_arys = np.array(all_arys)
    # all_arys = all_arys.max(axis=0)
    # # trim: remove consecutive 0s in the beginning and at the end
    # # sums = all_arys.sum(axis=1)
    # # ends = np.where(sums > 0)[0]

    # # sets max value to 1 and min value to 0
    # np.clip(all_arys, 0, 1, out=all_arys)

    # samples = []
    # for i in range(len(all_arys)):
    #     # print (i)
    #     sample_ix = int(i / num_notes)
    #     current_ix = (i - (sample_ix * num_notes))
    #     # print("Current IX", current_ix)
    #     if current_ix == 0:
    #         samples.append(
    #             np.zeros((samples_per_measure, num_notes), dtype=np.uint8))
    #     samples[sample_ix][current_ix] = all_arys[i]

    # samplelength = len(samples)
    # i = 0
    # while i < samplelength:
    #     sum_array = np.sum(samples[i])
    #     if sum_array <= 0:
    #         samples.pop(i)
    #         samplelength = samplelength-1
    #     else:
    #         i = i+1

    # # for i in range(len(samples)):
    # #     sum_array= np.sum(samples[i])
    # #     if sum_array<= 0:
    # #         samples.pop(i)
    # return samples

    has_time_sig = False
    flag_warning = False
    mid = MidiFile(fname)
    ticks_per_beat = mid.ticks_per_beat
    ticks_per_measure = 4 * ticks_per_beat

    for i, track in enumerate(mid.tracks):
        for msg in track:
            if msg.type == 'time_signature':
                new_tpm = msg.numerator * ticks_per_beat * 4 / msg.denominator
                if has_time_sig and new_tpm != ticks_per_measure:
                    flag_warning = True
                ticks_per_measure = new_tpm
                has_time_sig = True
    if flag_warning:
        print("  ^^^^^^ WARNING ^^^^^^")
        print("    " + fname)
        print("    Detected multiple distinct time signatures.")
        print("  ^^^^^^ WARNING ^^^^^^")
        return []

    all_notes = {}
    for i, track in enumerate(mid.tracks):
        abs_time = 0
        for msg in track:
            abs_time += msg.time
            if msg.type == 'note_on':
                if msg.velocity == 0:
                    continue
                note = msg.note - (128 - num_notes)/2
                assert(note >= 0 and note < num_notes)
                if note not in all_notes:
                    all_notes[note] = []
                else:
                    single_note = all_notes[note][-1]
                    if len(single_note) == 1:
                        single_note.append(single_note[0] + 1)
                all_notes[note].append(
                    [abs_time * samples_per_measure / ticks_per_measure])
            elif msg.type == 'note_off':
                if len(all_notes[note][-1]) != 1:
                    continue
                all_notes[note][-1].append(abs_time *samples_per_measure / ticks_per_measure)
    for note in all_notes:
        for start_end in all_notes[note]:
            if len(start_end) == 1:
                start_end.append(start_end[0] + 1)
    samples = []
    for note in all_notes:
        for start, end in all_notes[note]:
            sample_ix = int(start / samples_per_measure)
            while len(samples) <= sample_ix:
                samples.append(
                    np.zeros((samples_per_measure, num_notes), dtype=np.uint8))
            sample = samples[sample_ix]
            start_ix = int(start - sample_ix * samples_per_measure)
            if False:
                end_ix = min(end - sample_ix * samples_per_measure,
                             samples_per_measure)
                while start_ix < end_ix:
                    sample[start_ix, note] = 1
                    start_ix += 1
            else:
                sample[start_ix, int(note)] = 1
    return samples


def samples_to_midi(samples, fname, ticks_per_sample, thresh=0.5):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    ticks_per_beat = mid.ticks_per_beat
    ticks_per_measure = 4 * ticks_per_beat
    ticks_per_sample = ticks_per_measure / samples_per_measure
    abs_time = 0
    last_time = 0
    for sample in samples:
        for y in range(sample.shape[0]):
            abs_time += ticks_per_sample
            for x in range(sample.shape[1]):
                note = x + (128 - num_notes)/2
                if sample[y, x] >= thresh and (y == 0 or sample[y-1, x] < thresh):
                    delta_time = abs_time - last_time
                    track.append(Message('note_on', note=int(
                        note), velocity=127, time=int(delta_time)))
                    last_time = abs_time
                if sample[y, x] >= thresh and (y == sample.shape[0]-1 or sample[y+1, x] < thresh):
                    delta_time = abs_time - last_time
                    track.append(Message('note_off', note=int(
                        note), velocity=127, time=int(delta_time)))
                    last_time = abs_time
    mid.save(fname)


def arry2mid(ary, fname, tempo=500000):
    # get the difference
    new_ary = np.concatenate([np.array([[0] * 88]), np.array(ary)], axis=0)
    changes = new_ary[1:] - new_ary[:-1]
    # create a midi file with an empty track
    mid_new = mido.MidiFile()
    track = mido.MidiTrack()
    mid_new.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
    # add difference in the empty track
    last_time = 0
    for ch in changes:
        if set(ch) == {0}:  # no change
            last_time += 1
        else:
            on_notes = np.where(ch > 0)[0]
            on_notes_vol = ch[on_notes]
            off_notes = np.where(ch < 0)[0]
            first_ = True
            for n, v in zip(on_notes, on_notes_vol):
                new_time = last_time if first_ else 0
                track.append(mido.Message('note_on', note=n +
                                          21, velocity=v, time=new_time))
                first_ = False
            for n in off_notes:
                new_time = last_time if first_ else 0
                track.append(mido.Message('note_off', note=n +
                                          21, velocity=0, time=new_time))
                first_ = False
            last_time = 0
    mid_new.save(fname)
