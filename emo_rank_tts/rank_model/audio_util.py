import tgt
import torch
import numpy as np
import pyworld as pw
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram



def trim_audio(y, start_time, end_time, sampling_rate):
    s = int(np.round(start_time * sampling_rate))
    e = int(np.round(end_time * sampling_rate))
    return y[s:e].astype(np.float32)



def get_pitch(y, hop_length, sampling_rate):
    frame_period = hop_length / sampling_rate * 1000  # in milliseconds
    y = y.astype(np.float64)
    f0, t = pw.dio(y, sampling_rate, frame_period=frame_period)
    return pw.stonemask(y, f0, t, sampling_rate)



def get_mel(y, sampling_rate, hop_length, win_length, n_mels, n_fft, f_min, f_max):
    y = torch.FloatTensor(y)
    mel, energy = mel_spectogram(
        audio=y,
        sample_rate=sampling_rate,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        n_fft=n_fft,
        f_min=f_min,
        f_max=f_max,
        power=1,
        normalized=False,
        min_max_energy_norm=True,
        norm="slaney",
        mel_scale="slaney",
        compression=True
    )
    return mel, energy



def process_textgrid(textgrid_file, sampling_rate, hop_length, sil_phones):

    # read the TextGrid file and extract phone intervals
    tg = tgt.io.read_textgrid(textgrid_file, include_empty_intervals=True)
    tier = tg.get_tier_by_name('phones')
    intervals = [(iv.start_time, iv.end_time, iv.text or '') for iv in tier._objects]

    # filter out empty intervals and silence phones
    starts = np.array([s for s, e, p in intervals])
    ends = np.array([e for s, e, p in intervals])
    start_frames = np.round(starts * sampling_rate / hop_length).astype(int)
    end_frames = np.round(ends * sampling_rate / hop_length).astype(int)
    durations = end_frames - start_frames

    # filter out intervals with zero duration or silence phones
    labels = [p if p not in sil_phones else 'spn' for s, e, p in intervals]
    is_voiced = np.array([p not in sil_phones for s, e, p in intervals])
    if not is_voiced.any():
        print(f'No voiced phones found in {textgrid_file}')
        return [], np.array([], int), 0.0, 0.0

    # filter out intervals with zero duration
    first, last = np.where(is_voiced)[0][[0, -1]]
    phones = labels[first:last + 1]
    durations = durations[first:last + 1]
    speech_start = intervals[first][0]
    speech_end = intervals[last][1]

    return phones, durations, speech_start, speech_end



def expand(values, durations):
    return np.repeat(values, durations)