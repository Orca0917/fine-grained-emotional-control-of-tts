import os
import yaml
import torch
import numpy as np

from util import phoneme2sequence




class FastSpeech2Dataset(torch.utils.data.Dataset):

    def __init__(self, preprocessed_path, noise_symbol, speakers, emotions, mode='train'):
        super(FastSpeech2Dataset, self).__init__()
        
        self.preprocessed_path = preprocessed_path
        self.noise_symbol = noise_symbol
        self.speakers = speakers
        self.emotions = emotions

        self.data_paths = []
        with open(os.path.join(preprocessed_path, f'fs2_{mode}.txt'), 'r') as f:
            self.data_paths = [line.strip() for line in f.readlines()]
        
    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        data = np.load(data_path, allow_pickle=True)
        
        # Load features
        mel = data['mel']
        pitch = data['pitch']
        energy = data['energy']
        duration = data['durations']
        phoneme = data['phones'].tolist()

        # metadata
        speaker = data['speaker'].item()
        emotion = data['emotion'].item()
        text = data['transcript'].item().replace(self.noise_symbol.strip(), '').strip()
        audio_path = data['audio_path'].item()

        
        return {
            'mel': torch.FloatTensor(mel),
            'pitch': torch.FloatTensor(pitch),
            'energy': torch.FloatTensor(energy),
            'duration': torch.LongTensor(duration),
            'phoneme': torch.LongTensor(phoneme2sequence(phoneme)),
            'speaker': torch.tensor(self.speakers.index(speaker), dtype=torch.long),
            'emotion': torch.tensor(self.emotions.index(emotion), dtype=torch.long),
            'text': text,
            'audio_path': audio_path
        }



class TextMelCollateWithAlignment:

    def __call__(self, batch):

        # Right zero-pad all one-hot text sequences to the max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x['phoneme']) for x in batch]),
            dim=0, descending=True)

    
        max_input_len = input_lengths[0]

        phoneme_padded = torch.LongTensor(len(batch), max_input_len)
        phoneme_padded.zero_()
        duration_padded = torch.LongTensor(len(batch), max_input_len)
        duration_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            phoneme = batch[ids_sorted_decreasing[i]]['phoneme']
            phoneme_padded[i, :phoneme.size(0)] = phoneme
            duration = batch[ids_sorted_decreasing[i]]['duration']
            duration_padded[i, :duration.size(0)] = duration

        # Right zero-pad mel-spec
        num_mels = batch[0]['mel'].size(0)
        max_target_len = max([x['mel'].size(1) for x in batch])

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        pitch_padded = torch.FloatTensor(len(batch), max_target_len)
        pitch_padded.zero_()
        energy_padded = torch.FloatTensor(len(batch), max_target_len)
        energy_padded.zero_()
        rank_X = torch.FloatTensor(len(batch), num_mels + 2, max_target_len)
        rank_X.zero_()
        output_lengths = torch.LongTensor(len(batch))
        speakers = torch.LongTensor(len(batch))
        emotions = torch.LongTensor(len(batch))
        labels, wavs = [], []


        for i in range(len(ids_sorted_decreasing)):
            idx = ids_sorted_decreasing[i]
            mel = batch[idx]['mel']
            pitch = batch[idx]['pitch']
            energy = batch[idx]['energy']
            mel_padded[i, :, :mel.size(1)] = mel
            pitch_padded[i, :pitch.size(0)] = pitch
            energy_padded[i, :energy.size(0)] = energy
            output_lengths[i] = mel.size(1)
            labels.append(batch[idx]['text'])
            wavs.append(batch[idx]['audio_path'])
            speakers[i] = batch[idx]['speaker']
            emotions[i] = batch[idx]['emotion']

            rank_X[i, :, :mel.size(1)] = torch.cat(
                (mel, pitch.unsqueeze(0), energy.unsqueeze(0)), dim=0)

        mel_padded = mel_padded.permute(0, 2, 1)
        return (
            phoneme_padded,
            speakers,
            input_lengths,
            mel_padded,
            pitch_padded,
            energy_padded,
            duration_padded,
            output_lengths,
            labels,
            wavs,
            rank_X,
            emotions,
        )
    


if __name__ == '__main__':

    config = yaml.safe_load(open('parameter.yaml', 'r'))
    preprocessed_path   = config['path']['preprocessed_path']
    noise_symbol        = config['preprocessing']['noise_symbol']
    speakers            = config['preprocessing']['speakers']
    emotions            = config['preprocessing']['emotions']


    dataset = FastSpeech2Dataset(preprocessed_path, noise_symbol, 
                                 speakers, emotions, mode='train')
    collate_fn = TextMelCollateWithAlignment()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    for data in dataloader:
        phoneme, speaker, input_lengths, mel, pitch, energy, duration, output_lengths, labels, wavs = data

        print('Melspectrogram shape:    ', mel.shape)
        print('Pitch shape:             ', pitch.shape)
        print('Energy shape:            ', energy.shape)
        print('Duration shape:          ', duration.shape)
        print('Phoneme sequence:        ', phoneme.shape)
        print('*Total duration:         ', duration.sum(axis=1))
        print('Speaker index:           ', speaker)
        print('Text:                    ', labels)
        break
