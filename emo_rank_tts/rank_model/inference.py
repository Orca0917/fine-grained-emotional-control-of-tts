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
    speaker_list        = config['preprocessing']['speakers']
    emotion_list        = config['preprocessing']['emotions']
    preprocessed_path   = config['path']['preprocessed_path']

    n_mels              = config['audio']['n_mels']
    n_heads             = config['model']['n_heads']
    n_spk               = len(speaker_list)
    n_emo               = len(emotion_list)
    n_encoder_layers    = config['model']['n_encoder_layers']
    hidden_dim          = config['model']['hidden_dim']
    kernel_size         = config['model']['kernel_size']
    dropout             = config['model']['dropout']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FineGraindModel(n_mels, n_heads, n_spk, n_emo, n_encoder_layers,
                            hidden_dim, kernel_size, dropout)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)


    dataset = FineGrainedDataset(preprocessed_path, speaker_list, emotion_list, split='train')
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )

    intensity_storage = {spk: {emo: [] for emo in emotion_list} for spk in speaker_list}

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, dynamic_ncols=True):

            emo_X       = batch['emo_X'].to(device)
            neu_X       = batch['neu_X'].to(device)
            spk_ids     = batch['speakers'].to(device)
            emo_ids     = batch['emotions'].to(device)
            length      = batch['length'].to(device)
            B = emo_X.size(0)


            lambdas = torch.ones((2, B), device=device)
            _, _, I, _, _, _, r, _ = model(emo_X, neu_X, emo_ids, length, lambdas)

            # NumPy 변환
            I = I.detach().cpu().numpy()  # (B, T_mel, n_emotion)
            r = r.detach().cpu().numpy()  # (B)

            for i in range(B):
                spk = speaker_list[spk_ids[i].item()]
                emo = emotion_list[emo_ids[i].item()]
                T_mel = length[i].item()
                intensity_storage[spk][emo].append((r[i], I[i, :T_mel, :]))

    # prototype 저장용 배열
    intensity_representations = np.zeros(
        (n_spk, n_emo, bucket_size, n_emo), dtype=np.float32
    )

    # binning & 평균 계산
    for spk_idx, spk in enumerate(speaker_list):
        for emo_idx, emo in enumerate(emotion_list):
            vals = intensity_storage[spk][emo]

            if not vals:
                continue

            # score 기준 정렬
            vals.sort(key=lambda x: x[0])
            _, h_vals = zip(*vals)

            h_vals = np.concatenate(h_vals, axis=0)

            # 인덱스를 3등분
            splits = np.array_split(np.arange(len(h_vals)), bucket_size)
            for bin_idx, idxs in enumerate(splits):
                h_mean = h_vals[idxs].mean(axis=0)
                intensity_representations[spk_idx, emo_idx, bin_idx] = h_mean

    # 결과 저장
    np.save('intensity.npy', intensity_representations)
    print("Bucketization complete. Results saved to 'intensity.npy'.")


if __name__ == '__main__':

    config = yaml.safe_load(open('parameter.yaml'))
    # inference(config)

    bucketize(config)