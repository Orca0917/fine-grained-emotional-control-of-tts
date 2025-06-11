import os
import yaml
import torch
import torchaudio
import numpy as np
from itertools import product
from model import FastSpeech2
from util import text2sequence
from speechbrain.inference.vocoders import HIFIGAN


def get_intensity_rep(speaker, emotion, intensity_lv, T_phon, intensity_path):
    if emotion == 'neutral':
        intensity = torch.zeros(1, T_phon, 256, dtype=torch.float32)
    else:
        intensity_bank = np.load(intensity_path, allow_pickle=True)
        intensity = intensity_bank[speaker][emotion][intensity_lv]  # (H, )
        intensity = torch.from_numpy(intensity)
        intensity = intensity.reshape(1, 1, -1).expand(1, T_phon, -1)  # (B, T, H)

    return intensity


def load_model(fastspeech2_pth_path, model_config, speaker_list, device):

    model = FastSpeech2(**model_config, n_speakers=len(speaker_list))
    model.load_state_dict(torch.load(fastspeech2_pth_path, map_location=device))
    return model.to(device).eval()


def preprocess_input(text, speaker, emotion, speaker_list, emotion_list, device):
    phoneme = text2sequence(text.strip())
    spk_id = speaker_list.index(speaker)
    emo_id = emotion_list.index(emotion)
    phon_ids = torch.LongTensor(phoneme).unsqueeze(0).to(device)
    spkr_ids = torch.LongTensor([spk_id]).to(device)
    return phon_ids, spkr_ids, spk_id, emo_id


def inference(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    speaker_list                = config['preprocessing']['speakers']
    emotion_list                = config['preprocessing']['emotions']

    # process input
    text                        = config['inference']['text']
    bucket_size                 = config['inference']['bucket_size']
    rank_model_exp_name         = config['inference']['rank_model']
    fastspeech2_exp_name        = config['inference']['fastspeech2']
    exp_base_path               = config['path']['experiment_path']
    fastspeech2_config          = config['model']['fastspeech2']
    
    fastspeech2_pth_path        = os.path.join(exp_base_path, 'fastspeech2', fastspeech2_exp_name, 'best_model.pth')
    intensity_path              = os.path.join(exp_base_path, 'rank_model', rank_model_exp_name, 'intensity.npy')
    phoneme                     = text2sequence(text) 
    T_phon                      = len(phoneme) 

    # load FastSpeech2 model
    model = load_model(fastspeech2_pth_path, fastspeech2_config, speaker_list, device)
    vocoder = HIFIGAN.from_hparams(
        source="speechbrain/tts-hifigan-libritts-16kHz", 
        savedir=config['path']['vocoder_path']
    )

    # -- inference
    model.eval()
    preprocess_cache = {}
    for spk , emo in product(speaker_list, emotion_list):
        preprocess_cache[(spk, emo)] = preprocess_input(text, spk, emo, speaker_list, emotion_list, device)
        
    with torch.no_grad():
        for (spk, emo), (phon_ids, spkr_ids, spk_id, emo_id) in preprocess_cache.items():

            # prepare intensity representations
            intensities = [
                get_intensity_rep(spk_id, emo_id, lv, T_phon, intensity_path).to(device)
                for lv in range(bucket_size)
            ]

            # model inference
            for lv, intensity in enumerate(intensities):
                melspecs = model(phon_ids, spkr_ids, intensity=intensity)[0].permute(0, 2, 1)
                wav = vocoder.decode_batch(melspecs)
                torchaudio.save(f'/workspace/demo/{spk}_{emo}_{lv}.wav', wav.squeeze(1), 16000)


if __name__ == '__main__':

    config = yaml.safe_load(open('parameter.yaml'))
    inference(config)