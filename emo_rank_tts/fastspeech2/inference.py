import yaml
import torch
import torchaudio
import numpy as np
from model import FastSpeech2
from util import text2sequence, phoneme2sequence
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


def load_model(config, speaker_list, device):
    model = FastSpeech2(**config['model']['fastspeech2'], n_speakers=len(speaker_list))
    model.load_state_dict(torch.load(config['path']['model_path'], map_location=device))
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

    # load FastSpeech2 model
    model = load_model(config, speaker_list, device)
    vocoder = HIFIGAN.from_hparams(
        source="speechbrain/tts-hifigan-libritts-16kHz", 
        savedir=config['path']['vocoder_path']
    )

    # process input
    text                        = config['inference']['text']
    speaker                     = config['inference']['speaker']
    emotion                     = config['inference']['emotion']
    intensity_lv                = config['inference']['intensity_level']
    
    phon_ids, spkr_ids, spk_id, emo_id = preprocess_input(text, speaker, emotion, speaker_list, emotion_list, device)
    T_phon = phon_ids.size(1)
    intensity = get_intensity_rep(spk_id, emo_id, intensity_lv, T_phon, config['path']['intensity_path']).to(device)

    # -- inference
    model.eval()
    with torch.no_grad():
        melspecs = model(phon_ids, spkr_ids, intensity=intensity)[0].permute(0, 2, 1)
        wav = vocoder.decode_batch(melspecs)
        torchaudio.save('result.wav', wav.squeeze(1), 16000)


if __name__ == '__main__':

    config = yaml.safe_load(open('parameter.yaml'))
    inference(config)