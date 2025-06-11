import os
import yaml
import torch
import numpy as np
from tqdm import tqdm

from rank_model.model import RankModel
from rank_model.dataset import FineGrainedDataset, collate_fn


def bucketize(config) -> None:
    """
    Compute intensity prototypes by binning per-speaker-and-emotion representations.
    Results are saved as a NumPy array of shape
    (num_speakers, num_emotions, bucket_size, num_emotions).
    """

    # Load configuration
    exp_base_path       = config['path']['experiment_path']
    exp_name            = config['inference']['exp_name']
    bucket_size         = config['inference']['bucket_size']
    speaker_list        = config['preprocessing']['speakers']
    emotion_list        = config['preprocessing']['emotions']
    preprocessed_path   = config['path']['preprocessed_path']

    # Hyperparameters
    n_mels              = config['audio']['n_mels']
    n_heads             = config['model']['n_heads']
    n_spk               = len(speaker_list)
    n_emo               = len(emotion_list)
    n_encoder_layers    = config['model']['n_encoder_layers']
    hidden_dim          = config['model']['hidden_dim']
    kernel_size         = config['model']['kernel_size']
    dropout             = config['model']['dropout']


    # Paths and device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    exp_path = os.path.join(exp_base_path, 'rank_model', exp_name)
    best_pth_path = os.path.join(exp_path, 'best_model.pth')
    output_path = os.path.join(exp_path, 'intensity.npy')

    print(f"- Using device: {device}")
    print(f"- Using best model from {exp_name}")


    # model
    model = RankModel(n_mels, n_heads, n_emo, n_encoder_layers, hidden_dim, kernel_size, dropout)
    model.load_state_dict(torch.load(best_pth_path, map_location=device))
    model = model.to(device)


    # dataset
    dataset = FineGrainedDataset(preprocessed_path, speaker_list, emotion_list, split='train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=4)


    # Initialize storage for intensity sequences
    intensity_storage = {spk: {emo: [] for emo in emotion_list} for spk in speaker_list}


    # Inference
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, dynamic_ncols=True):

            emo_X       = batch['emo_X'].to(device)
            neu_X       = batch['neu_X'].to(device)
            spk_ids     = batch['speakers'].to(device)
            emo_ids     = batch['emotions'].to(device)
            length      = batch['length'].to(device)
            B           = emo_X.size(0)
            lambdas     = torch.ones((2, B), device=device)

            # Forward pass: retrieve intensity I and relevance score r
            _, _, I, _, h, _, r, _ = model(emo_X, neu_X, emo_ids, length, lambdas)

            # Move to CPU for aggregation
            I = I.detach().cpu().numpy()  # (B, T_mel, n_emotion)
            h = h.detach().cpu().numpy()  # (B, n_emotion)
            r = r.detach().cpu().numpy()  # (B)

            for i in range(B):
                spk = speaker_list[spk_ids[i].item()]
                emo = emotion_list[emo_ids[i].item()]
                T_mel = length[i].item()

                # Append tuple of (score, intensity sequence)
                intensity_storage[spk][emo].append((r[i], I[i, :T_mel, :]))


    # Initialize prototypes array
    prototypes = np.zeros(
        (n_spk, n_emo, bucket_size, n_emo), dtype=np.float32
    )


    # Binning and averaging
    for si, spk in enumerate(speaker_list):
        for ei, emo in enumerate(emotion_list):
            entries = intensity_storage[spk][emo]
            if not entries:
                continue

            # Sort by score r
            entries.sort(key=lambda x: x[0])
            _, intensities = zip(*entries)

            all_feats = np.concatenate(intensities, axis=0)

            # Split indices into equal buckets
            splits = np.array_split(np.arange(len(all_feats)), bucket_size)
            for bi, idxs in enumerate(splits):
                prototypes[si, ei, bi] = all_feats[idxs].mean(axis=0)


    # Save results
    np.save(output_path, prototypes)
    print(f"- Bucketization complete. Results saved to {output_path}.")


if __name__ == '__main__':
    config = yaml.safe_load(open('parameter.yaml'))
    bucketize(config)