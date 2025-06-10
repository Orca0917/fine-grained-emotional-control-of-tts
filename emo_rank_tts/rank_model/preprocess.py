import os
import json
import yaml
import scipy
import random
import librosa
import numpy as np

from glob import glob
from tqdm import tqdm
from pathlib import Path
from audio_util import get_pitch, get_mel, trim_audio, process_textgrid, expand
from sklearn.preprocessing import StandardScaler


def _average_by_duration(values, durations):
    out = np.zeros(len(durations), dtype=np.float32)
    idx = 0
    for i, d in enumerate(durations):
        if d > 0:
            out[i] = values[idx:idx + d].mean()
        idx += max(d, 0)
    return out



def remove_outliers(x):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    mask = (x >= (q1 - 1.5 * iqr)) & (x <= (q3 + 1.5 * iqr))
    return x[mask]



def normalize_field(preprocessed_path, speaker, emotion, field, mean, std):
    paths = glob(os.path.join(preprocessed_path, speaker, f'{emotion}_*.npz'))
    global_min, global_max = np.inf, -np.inf

    for fp in paths:
        arr = dict(np.load(fp))
        arr[field] = (arr[field] - mean) / std
        np.savez(fp, **arr)
        global_min = min(global_min, arr[field].min())
        global_max = max(global_max, arr[field].max())

    return float(global_min), float(global_max)



def feature_extraction(
        corpus_path, 
        textgrid_path,
        preprocessed_path, 
        speaker, 
        emotion,
        sampling_rate,
        hop_length,
        win_length, 
        n_mels, 
        n_fft, 
        f_min, 
        f_max,
        noise_symbol,
        sil_phones,
        pitch_averaging,
        energy_averaging,
        tbar
    ):

    pitch_scaler = StandardScaler()
    energy_scaler = StandardScaler()
    
    wav_paths = glob(os.path.join(corpus_path, speaker, f'{emotion}_*.wav'))
    for audio_path in wav_paths:

        tbar.update(1)

        audio_id = Path(audio_path).stem.split('_')[-1]
        tgt_path = os.path.join(textgrid_path, speaker, f'{emotion}_{audio_id}.TextGrid')
        transcript_path = Path(os.path.join(corpus_path, speaker, f'{emotion}_{audio_id}.lab'))

        # check the path existence
        if not os.path.exists(tgt_path):
            continue
        
        phones, durations, start_t, end_t = process_textgrid(tgt_path, sampling_rate, hop_length, sil_phones)

        if start_t >= end_t:
            print(f"Invalid start/end: {audio_path}")
            continue

        # trim audio
        y, sr = librosa.load(audio_path, sr=sampling_rate)
        y = trim_audio(y, start_t, end_t, sampling_rate)

        # transcript
        transcript = transcript_path.read_text().strip().replace(noise_symbol, '')

        # 1. pitch
        pitch = get_pitch(y, hop_length, sampling_rate)
        if np.count_nonzero(pitch) <= 1:
            print(f"Invalid pitch: {audio_path}")
            continue
        pitch = pitch[:sum(durations)]

        non_zero_ids = np.where(pitch != 0)[0]
        interp_fn = scipy.interpolate.interp1d(
            non_zero_ids, pitch[non_zero_ids],
            fill_value=(pitch[non_zero_ids[0]], pitch[non_zero_ids[-1]]),
            bounds_error=False
        )
        pitch = interp_fn(np.arange(0, len(pitch)))
        
        # 2. melspectrogram, energy
        mel, energy = get_mel(y, sampling_rate, hop_length, win_length, n_mels, n_fft, f_min, f_max)
        mel, energy = mel.numpy(), energy.numpy()
        mel = mel[:, :sum(durations)]
        energy = energy[:sum(durations)]

        if pitch_averaging:
            pitch = _average_by_duration(pitch, durations)
            pitch = expand(pitch, durations)

        if energy_averaging:
            energy = _average_by_duration(energy, durations)
            energy = expand(energy, durations)

        pitch_clean = remove_outliers(pitch)
        energy_clean = remove_outliers(energy)
        pitch_scaler.partial_fit(pitch_clean.reshape((-1, 1)))
        energy_scaler.partial_fit(energy_clean.reshape((-1, 1)))

        assert len(mel[0]) == len(pitch) == len(energy)
        np.savez(
            os.path.join(preprocessed_path, speaker, f'{emotion}_{audio_id}.npz'),
            
            # metadata
            phones=phones,
            emotion=emotion,
            speaker=speaker,
            audio_id=audio_id,
            audio_path=audio_path,
            transcript=transcript,
            textgrid_path=tgt_path,
            
            # inputs
            mel=mel,
            pitch=pitch,
            energy=energy,
            durations=durations,
        )

    # calculate the normalization parameters
    p_mean, p_std = pitch_scaler.mean_[0], pitch_scaler.scale_[0]
    e_mean, e_std = energy_scaler.mean_[0], energy_scaler.scale_[0]

    # normalize the fields and save the stats
    p_min, p_max = normalize_field(preprocessed_path, speaker, emotion, 'pitch', p_mean, p_std)
    e_min, e_max = normalize_field(preprocessed_path, speaker, emotion, 'energy', e_mean, e_std)

    # save the stats
    stats_file = Path(preprocessed_path, 'stats.json')
    stats = json.loads(stats_file.read_text()) if stats_file.exists() else {}
    stats.setdefault(speaker, {})[emotion] = {
        'pitch': [p_min, p_max, p_mean, p_std],
        'energy': [e_min, e_max, e_mean, e_std],
    }
    stats_file.write_text(json.dumps(stats, indent=4))



def prepare_data_lists(
        preprocessed_path,
        speakers,
        emotions,
        match_transcript,
):
    train_list = []
    test_list = []

    for speaker in speakers:

        # neutral audio ids per speaker
        neu_paths = glob(os.path.join(preprocessed_path, speaker, f'neutral_*.npz'))
        neu_ids = [os.path.basename(neu_path)[:-4].split('_')[-1] for neu_path in neu_paths]
        neu_ids = sorted(neu_ids)

        for emotion in emotions:

            if emotion == 'neutral':
                continue

            emo_paths = glob(os.path.join(preprocessed_path, speaker, f'{emotion}_*.npz'))
            emo_ids = [os.path.basename(emo_path)[:-4].split('_')[-1] for emo_path in emo_paths]
            emo_ids = sorted(emo_ids)

            # find common audio ids
            if match_transcript:
                neu_ids = set(neu_ids)
                emo_ids = set(emo_ids)
                audio_ids = sorted(list(neu_ids.intersection(emo_ids)))

                # train
                for audio_id in audio_ids[:-5]:
                    train_list.append('|'.join([speaker, emotion, audio_id, audio_id]))

                # test
                for audio_id in audio_ids[-5:]:
                    test_list.append('|'.join([speaker, emotion, audio_id, audio_id]))

            else:

                # train
                for emo_audio_id in emo_ids[:-5]:
                    for neu_audio_id in random.sample(neu_ids, k=10):
                        train_list.append('|'.join([speaker, emotion, emo_audio_id, neu_audio_id]))

                # test
                for emo_audio_id in emo_ids[-5:]:
                    for neu_audio_id in random.sample(neu_ids, k=10):
                        test_list.append('|'.join([speaker, emotion, emo_audio_id, neu_audio_id]))


    print(f'Length of train list: {len(train_list)}')
    print(f'Length of test list: {len(test_list)}')


    with open(os.path.join(preprocessed_path, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_list) + '\n')
    with open(os.path.join(preprocessed_path, 'test.txt'), 'w') as f:
        f.write('\n'.join(test_list) + '\n')



if __name__ == '__main__':
    config = yaml.safe_load(open('parameter.yaml'))
    data_path           = config['path']['data_path']
    corpus_path         = config['path']['corpus_path']
    textgrid_path       = config['path']['textgrid_path']
    preprocessed_path   = config['path']['preprocessed_path']

    sampling_rate       = config['audio']['sampling_rate']
    hop_length          = config['audio']['hop_length']
    win_length          = config['audio']['win_length']
    n_mels              = config['audio']['n_mels']
    n_fft               = config['audio']['n_fft']
    f_min               = config['audio']['f_min']
    f_max               = config['audio']['f_max']

    speakers            = config['preprocessing']['speakers']
    emotions            = config['preprocessing']['emotions']
    noise_symbol        = config['preprocessing']['noise_symbol']
    sil_phones          = config['preprocessing']['sil_phones']
    pitch_averaging     = config['preprocessing']['pitch_averaging']
    energy_averaging    = config['preprocessing']['energy_averaging']
    match_transcript    = config['preprocessing']['match_transcript']

    total_wavs = len(glob(os.path.join(corpus_path, '*', '*.wav')))
    tbar = tqdm(total=total_wavs, desc='Processing audio files', dynamic_ncols=True)

    # feature extraction
    for speaker in speakers:
        for emotion in emotions:

            # check the path existence
            if not os.path.exists(os.path.join(data_path, speaker, emotion)):
                continue

            # preprocessed path
            os.makedirs(os.path.join(preprocessed_path, speaker), exist_ok=True)

            # mel, energy, pitch, durations
            feature_extraction(
                corpus_path, textgrid_path, preprocessed_path, speaker, emotion, sampling_rate, 
                hop_length, win_length, n_mels, n_fft, f_min, f_max, noise_symbol, sil_phones,
                pitch_averaging, energy_averaging, tbar)

    tbar.close()


    # create train, valid datasets
    prepare_data_lists(preprocessed_path, speakers, emotions, match_transcript)