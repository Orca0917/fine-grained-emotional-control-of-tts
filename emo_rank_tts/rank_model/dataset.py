import os
import yaml
import torch
import numpy as np
from util import plot_melspectrograms


class FineGrainedDataset(torch.utils.data.Dataset):

    def __init__(self, preprocessed_path, speakers, emotions, split='train'):
        self.split = split
        self.preprocessed_path = preprocessed_path

        self.emo_audio_id = []
        self.neu_audio_id = []
        self.speakers = speakers
        self.emotions = emotions
        self.speaker = []
        self.emotion = []

        with open(os.path.join(preprocessed_path, f'{split}.txt'), 'r') as f:
            for line in f.readlines():
                speaker, emotion, emo_audio_id, neu_audio_id = line.strip().split('|')
                self.emo_audio_id.append(emo_audio_id)
                self.neu_audio_id.append(neu_audio_id)
                self.speaker.append(speaker)
                self.emotion.append(emotion)

    def __len__(self):
        return len(self.emo_audio_id)
    
    def __getitem__(self, idx):
        emo_audio_id = self.emo_audio_id[idx]
        neu_audio_id = self.neu_audio_id[idx]
        speaker = self.speaker[idx]
        emotion = self.emotion[idx]

        # load preprocessed data
        speaker_base_path = os.path.join(self.preprocessed_path, speaker)
        emo_data = dict(np.load(os.path.join(speaker_base_path, f'{emotion}_{emo_audio_id}.npz')))
        neu_data = dict(np.load(os.path.join(speaker_base_path, f'neutral_{neu_audio_id}.npz')))

        # prepare inputs
        emo_X = self._prepare_X(emo_data)
        neu_X = self._prepare_X(neu_data)
        speaker_id = self.speakers.index(speaker)
        emotion_id = self.emotions.index(emotion)

        return {
            'emo_X': emo_X,
            'neu_X': neu_X,
            'speaker': speaker_id,
            'emotion': emotion_id,
        }
    
    def _prepare_X(self, data):
        mel             = data['mel']
        pitch           = data['pitch']
        energy          = data['energy']
        durations       = data['durations']
        phones          = data['phones']
        transcript      = data['transcript']
        audio_path      = data['audio_path']
        textgrid_path   = data['textgrid_path']


        X = np.concatenate([mel, pitch.reshape((1, -1)), energy.reshape((1, -1))], axis=0)
        return X
    

def collate_fn(batch):

    B = len(batch)
    n_mels = batch[0]['emo_X'].shape[0] - 2

    max_T = max([item['emo_X'].shape[1] for item in batch] \
                + [item['neu_X'].shape[1] for item in batch])


    # create a tensor to hold the mel spectrograms
    length = torch.LongTensor(B)
    emo_X = np.zeros((B, n_mels + 2, max_T), dtype=np.float32)  # +2 for pitch and energy
    neu_X = np.zeros((B, n_mels + 2, max_T), dtype=np.float32)  # +2 for pitch and energy
    speakers = torch.LongTensor([item['speaker'] for item in batch])
    emotions = torch.LongTensor([item['emotion'] for item in batch])


    # pad the mel spectrograms to the same length
    for idx in range(B):

        _emo_X = batch[idx]['emo_X']
        _neu_X = batch[idx]['neu_X']
        T = min(_emo_X.shape[1], _neu_X.shape[1])

        # truncate to the minimum length
        _emo_X = _emo_X[:, :T]
        _neu_X = _neu_X[:, :T]

        # pad the mel spectrograms to the maximum length
        pad_width = max_T - T
        emo_X[idx] = np.pad(_emo_X, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        neu_X[idx] = np.pad(_neu_X, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        length[idx] = T

    # convert to tensors
    emo_X = torch.FloatTensor(emo_X).permute(0, 2, 1)  # (B, T, n_mels + 2)
    neu_X = torch.FloatTensor(neu_X).permute(0, 2, 1)  # (B, T, n_mels + 2)

    return {
        'emo_X': emo_X,
        'neu_X': neu_X,
        'speakers': speakers,
        'emotions': emotions,
        'length': length,
    }


if __name__ == '__main__':

    config = yaml.safe_load(open('config.yaml'))
    preprocessed_path   = config['path']['preprocessed_path']
    n_mels              = config['preprocessing']['n_mels']
    speakers            = config['preprocessing']['speakers']
    emotions            = config['preprocessing']['emotions']

    dataset = FineGrainedDataset(preprocessed_path, speakers, emotions, split='train')
    print(f"Dataset size: {len(dataset)}")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    for batch in dataloader:
        emo_X = batch['emo_X']
        neu_X = batch['neu_X']
        speakers = batch['speakers']
        emotions = batch['emotions']
        length = batch['length']

        print(f"Batch size: {len(emo_X)}")
        print(f"Speakers: {speakers}")
        print(f"Emotions: {emotions}")
        print(f"Lengths: {length}")

        plot_melspectrograms(emo_X[0][:n_mels, :], neu_X[0][:n_mels, :], 
                             save_path='./sample_dataset_mel.png')

        break
