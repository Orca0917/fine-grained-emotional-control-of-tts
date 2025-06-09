import os
import yaml
import librosa
from tqdm.rich import tqdm
from glob import glob
from speechbrain.utils.text_to_sequence import _clean_text
from scipy.io.wavfile import write


def prepare_audio_transcript_pairs(data_path, noise_symbol):

    audio_id_to_transcript = {}
    with open(os.path.join(data_path, 'cmuarctic.data')) as f:
        for line in f.readlines():
            audio_id, transcript = line[2:-2].split('\"')[:2]

            audio_id = audio_id.strip()
            transcript = transcript.strip()

            if audio_id.startswith('arctic_b'):
                continue
            
            audio_id = audio_id[-4:]
            transcript = noise_symbol + _clean_text(transcript, ['english_cleaners']) + noise_symbol

            audio_id_to_transcript[audio_id] = transcript.strip()
    
    return audio_id_to_transcript



def prepare_mfa(data_path, corpus_path, speakers, emotions, sr, transcript):

    for speaker in tqdm(speakers):
        for emotion in emotions:

            # check the path existence: josh has only three emotions
            spk_emo_path = os.path.join(data_path, speaker, emotion)
            if not os.path.exists(spk_emo_path):
                continue
            
            # resample and create .lab file
            for wav_path in glob(os.path.join(spk_emo_path, '*.wav')):

                y, sr = librosa.load(wav_path, sr=sr)

                audio_id = os.path.basename(wav_path)[-8:-4]
                text = transcript[audio_id]

                os.makedirs(os.path.join(corpus_path, speaker), exist_ok=True)

                tgt_path = os.path.join(corpus_path, speaker, f'{emotion}_{audio_id}')
                write(tgt_path + '.wav', sr, y)

                with open(tgt_path + '.lab', 'w') as f:
                    f.write(text + '\n')



if __name__ == '__main__':

    # Load configuration
    config = yaml.safe_load(open('parameter.yaml'))

    data_path       = config['path']['data_path']
    corpus_path     = config['path']['corpus_path']
    sampling_rate   = config['audio']['sampling_rate']
    speakers        = config['preprocessing']['speakers']
    emotions        = config['preprocessing']['emotions']
    noise_symbol    = config['preprocessing']['noise_symbol']

    # Prepare audio-transcript pairs
    audio_id_to_transcript = prepare_audio_transcript_pairs(data_path, noise_symbol)
    
    # Prepare corpus for Montreal Forced Aligner
    if os.path.exists(corpus_path):
        print('Corpus path already exists, skipping preparation.')
    else:
        print('Preparing corpus for Montreal Forced Aligner...')
        prepare_mfa(data_path, corpus_path, speakers, emotions,
                    sampling_rate, audio_id_to_transcript)
        print('Corpus preparation completed.')