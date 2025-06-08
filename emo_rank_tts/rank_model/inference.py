import json
import yaml
import torch

import numpy as np
from tqdm import tqdm
from model import FineGraindModel
from dataset import FineGrainedDataset, collate_fn

def inference(config):
    
    model_path = config['inference']['best_pth_path']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(model_path, map_location=device)
    model.eval()

    print(f'Using device: {device}')
    print(f'Loaded model from {model_path}')

    pass


def bucketize(config):

    model_path          = config['inference']['best_pth_path']
    bucket_size         = config['inference']['bucket_size']
    speakers            = config['preprocessing']['speakers']
    emotions            = config['preprocessing']['emotions']
    preprocessed_path   = config['path']['preprocessed_path']

    n_mels              = config['audio']['n_mels']
    n_heads             = config['model']['n_heads']
    n_speakers          = len(speakers)
    n_emotions          = len(emotions)
    n_encoder_layers    = config['model']['n_encoder_layers']
    hidden_dim          = config['model']['hidden_dim']
    kernel_size         = config['model']['kernel_size']
    dropout             = config['model']['dropout']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FineGraindModel(n_mels, n_heads, n_speakers, n_emotions, n_encoder_layers,
                            hidden_dim, kernel_size, dropout)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)


    dataset = FineGrainedDataset(preprocessed_path, speakers, emotions, split='train')
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )

    intensity_representations = {speaker: {emotion: [] for emotion in emotions} for speaker in speakers}

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, dynamic_ncols=True):

            emo_X = batch['emo_X'].to(device)
            neu_X = batch['neu_X'].to(device)
            speaker = batch['speakers'].to(device)
            emotion = batch['emotions'].to(device)
            length = batch['length'].to(device)
            targets = (emotion, torch.full_like(emotion, 0, device=device))

            B = emo_X.size(0)
            lambda_i = torch.ones(B, device=device)
            lambda_j = torch.zeros(B, device=device)
            lambdas = torch.stack([lambda_i, lambda_j], dim=0)  # (2, B)

            # forward pass
            predictions = model(emo_X, neu_X, emotion, length, lambdas)
            _, _, h, _, _, _, r, _ = predictions

            r = r.squeeze(-1)
            h = h.detach().cpu().numpy()

            for i in range(B):
                speaker_name = speakers[speaker[i].item()]
                emotion_name = emotions[emotion[i].item()]
                intensity_representations[speaker_name][emotion_name].append((r[i].item(), h[i]))


    # speaker의 emotion별로 (r, h) 쌍을 r을 기준으로 3개의 bin으로 만들고 각 bin의 평균 h를 구함
    result = {speaker: {emotion: [] for emotion in emotions} for speaker in speakers}
    for speaker in speakers:
        for emotion in emotions:
            values = intensity_representations[speaker][emotion]
            if not values:
                continue

            # r 값을 기준으로 정렬
            values.sort(key=lambda x: x[0])
            r_values, h_values = zip(*values)

            # 3개의 bin으로 나누기
            bin_size = len(r_values) // bucket_size
            h_means = [sum(h_values[i:i + bin_size]) / bin_size for i in range(0, len(h_values), bin_size)]

            result[speaker][emotion] = h_means[:bucket_size]


    # npz 저장
    np.savez('result.npz', intensity=result)
    print("Bucketization complete. Results saved to 'result.npz'.")


if __name__ == '__main__':

    config = yaml.safe_load(open('parameter.yaml'))
    # inference(config)

    bucketize(config)