import yaml
import torch

from model import RankModel
from dataset import FineGrainedDataset

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
    preprocessed_path   = config['paths']['preprocessed_path']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(model_path, map_location=device)

    dataset = FineGrainedDataset(preprocessed_path, speakers, emotions, split='train')
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=4,
    )

    intensity_representations = {speaker: {emotion: [] for emotion in emotions} for speaker in speakers}

    model.eval()
    with torch.no_grad():
        for batch in dataloader:

            emo_X = batch['emo_X'].to(device)
            neu_X = batch['neu_X'].to(device)
            speaker = batch['speakers'].to(device)
            emotion = batch['emotions'].to(device)
            length = batch['length'].to(device)
            targets = (emotion, torch.full_like(emotion, 0, device=device))

            B = emo_X.size(0)
            lambda_i = torch.ones(B, device=device)
            lambda_j = torch.zeros(B, device=device)
            lambdas = torch.stack([lambda_i, lambda_j], dim=1)  # (B, 2)

            # forward pass
            predictions = model(emo_X, neu_X, speaker, length, targets, lambdas)
            _, _, h, _, _, _, r, _ = predictions

            r = r.squeeze(-1)

            for i in range(B):
                speaker_name = speakers[speaker[i].item()]
                emotion_name = emotions[emotion[i].item()]
                intensity_representations[speaker_name][emotion_name].append((r[i].item(), h[i].item()))


    # speaker의 emotion별로 (r, h) 쌍을 r을 기준으로 3개의 bin으로 만들고 각 bin의 평균 h를 구함
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
            bins = [r_values[i:i + bin_size] for i in range(0, len(r_values), bin_size)]
            h_means = [sum(h_values[i:i + bin_size]) / bin_size for i in range(0, len(h_values), bin_size)]

            print(f'Speaker: {speaker}, Emotion: {emotion}, Bins: {bins}, H Means: {h_means}')
            


if __name__ == '__main__':

    config = yaml.safe_load(open('config.yaml'))
    # inference(config)

    bucketize(config)