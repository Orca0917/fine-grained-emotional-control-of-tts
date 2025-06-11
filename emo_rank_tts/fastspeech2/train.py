import os
import yaml
import torch
from tqdm import tqdm
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from speechbrain.inference.vocoders import HIFIGAN

from rank_model.model import RankModel
from fastspeech2.loss import Loss
from fastspeech2.model import FastSpeech2
from fastspeech2.dataset import FastSpeech2Dataset, TextMelCollateWithAlignment
from fastspeech2.util import batch_to_device, plot_fastspeech2_melspecs, increment_path, synthesize_sample


def get_intensity_representation(intensity_extractor, batch, device):

    (
        phoneme, _, phon_len, _, _, _,
        duration_tgt, mel_len, _, _, rank_X, emo_ids
    ) = batch

    with torch.no_grad():
        B = rank_X.size(0)
        lambdas = torch.ones((2, B), device=device)

        I = intensity_extractor(rank_X, mel_len, emo_ids)

        B, T_phon_max = phoneme.shape
        _, _, D = I.shape
        intensity_rep = torch.zeros((B, T_phon_max, D), device=device)

        for b in range(B):

            T_phon = phon_len[b].item()
            durations = duration_tgt[b].long()[:T_phon]
            T_mel = int(durations.sum().item())

            I_b = I[b, :T_mel, :]

            phon_idx = torch.repeat_interleave(
                torch.arange(T_phon, device=device), durations
            )

            sum_rep = torch.zeros((T_phon, D), device=device)
            sum_rep.index_add_(0, phon_idx, I_b)

            denom = durations.unsqueeze(1).float().clamp(min=1.0)
            intensity_rep[b, :T_phon, :] = sum_rep / denom

    return intensity_rep


def train_one_epoch(dataloader, model, intensity_extractor, criterion, optim, device, epoch, exp_path, writer):
    model.train()
    
    pbar = tqdm(dataloader, desc=f'Train Epoch {epoch + 1}', unit='batch', dynamic_ncols=True)
    epoch_avg_loss = defaultdict(float)

    for idx, batch in enumerate(pbar):

        batch = batch_to_device(batch, device)
        (
            phoneme, spk_ids, phon_len, mel_tgt, pitch_tgt, energy_tgt,
            duration_tgt, mel_len, _, _, _, _
        ) = batch

        # intensity extraction
        intensity_rep = get_intensity_representation(intensity_extractor, batch, device)
        
        # forward pass
        predictions = model(phoneme, spk_ids, duration_tgt, pitch_tgt, energy_tgt, intensity=intensity_rep)

        # compute loss
        targets = (mel_tgt, duration_tgt, pitch_tgt, energy_tgt, mel_len, phon_len)
        loss = criterion(predictions, targets, epoch)

        # backward pass
        optim.zero_grad()
        loss['total_loss'].backward()
        optim.step()

        # accumulate loss
        for loss_name, loss_value in loss.items():
            epoch_avg_loss[loss_name] += loss_value.item()

        if idx == 0 and epoch % 10 == 0:
            melspecs = predictions[0].cpu().detach().numpy()
            y_melspecs = mel_tgt.cpu().detach().numpy()
            plot_fastspeech2_melspecs(melspecs, y_melspecs, epoch, exp_path)

        pbar.set_postfix(
            total_loss="{:.04f}".format(loss['total_loss'].item()),
            mel_loss="{:.04f}".format(loss['mel_loss'].item()),
            pitch_loss="{:.04f}".format(loss['pitch_loss'].item()),
            dur_loss="{:.04f}".format(loss['dur_loss'].item()),
        )
    
    pbar.close()

    # tensorboard logging
    epoch_avg_loss = {k: v / len(dataloader) for k, v in epoch_avg_loss.items()}
    for k, v in epoch_avg_loss.items():
        writer.add_scalar(f'Loss/{k}', v, epoch)

    # save model
    torch.save(model.state_dict(), f"{exp_path}/last_model.pth")
    
    return epoch_avg_loss['total_loss']


def valid_one_epoch(dataloader, model, rank_model, vocoder, criterion, device, epoch, exp_path, writer):

    model.eval()
    pbar = tqdm(dataloader, desc=f'Valid Epoch {epoch + 1}', unit='batch', dynamic_ncols=True)

    epoch_avg_loss = defaultdict(float)
    with torch.no_grad():
        for idx, batch in enumerate(pbar):

            batch = batch_to_device(batch, device)
            (
                phoneme, spk_ids, phon_len, mel_tgt, pitch_tgt, energy_tgt,
                duration_tgt, mel_len, _, _, _, _
            ) = batch

            # intensity extraction
            intensity_rep = get_intensity_representation(rank_model, batch, device)

            # forward pass
            predictions = model(phoneme, spk_ids, duration_tgt, pitch_tgt, energy_tgt, intensity=intensity_rep)

            # compute loss
            targets = (mel_tgt, duration_tgt, pitch_tgt, energy_tgt, mel_len, phon_len)
            loss = criterion(predictions, targets, epoch)

            # accumulate loss
            for loss_name, loss_value in loss.items():
                epoch_avg_loss[loss_name] += loss_value

            if idx == 0 and epoch % 10 == 0:
                melspecs = predictions[0]
                y_melspecs = mel_tgt
                synthesize_sample(vocoder, melspecs, y_melspecs, mel_len, exp_path, epoch)

                melspecs = melspecs.cpu().detach().numpy()
                y_melspecs = y_melspecs.cpu().detach().numpy()
                plot_fastspeech2_melspecs(melspecs, y_melspecs, epoch, exp_path, train=False)
                

            pbar.set_postfix(
                total_loss="{:.04f}".format(loss['total_loss'].item()),
                mel_loss="{:.04f}".format(loss['mel_loss'].item()),
                pitch_loss="{:.04f}".format(loss['pitch_loss'].item()),
                dur_loss="{:.04f}".format(loss['dur_loss'].item()),
            )

    pbar.close()

    # tensorboard logging
    epoch_avg_loss = {k: v / len(dataloader) for k, v in epoch_avg_loss.items()}
    for k, v in epoch_avg_loss.items():
        writer.add_scalar(f'Valid/Loss/{k}', v, epoch)

    return epoch_avg_loss['total_loss']


def train(config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths and configurations
    preprocessed_path       = config['path']['preprocessed_path']
    exp_base_path           = config['path']['experiment_path']
    vocoder_path            = config['path']['vocoder_path']
    noise_symbol            = config['preprocessing']['noise_symbol']
    speaker_list            = config['preprocessing']['speakers']
    emotion_list            = config['preprocessing']['emotions']

    # Hyperparameters
    lr                      = config['train']['learning_rate']
    batch_size              = config['train']['batch_size']
    n_epochs                = config['train']['n_epochs']
    max_iterations          = config['train']['max_iterations']
    max_patience            = config['train']['patience']
    fastspeech2_config      = config['model']['fastspeech2']
    rank_model_config       = config['model']['rank_model']
    n_mels                  = config['audio']['n_mels']
    rank_model_exp_name     = config['inference']['rank_model']
    loss_config             = config['loss']


    n_speakers              = len(speaker_list)
    n_emotions              = len(emotion_list)
    rank_model_pth_path     = os.path.join('/workspace/experiments/rank_model', rank_model_exp_name, 'best_model.pth')


    # Load dataset
    collate_fn = TextMelCollateWithAlignment()
    train_dataset = FastSpeech2Dataset(preprocessed_path, noise_symbol, speaker_list, emotion_list, mode='train')
    valid_dataset = FastSpeech2Dataset(preprocessed_path, noise_symbol, speaker_list, emotion_list, mode='valid')
    
    # dataloaoder
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True, drop_last=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True, drop_last=False
    )

    # model
    fastspeech2 = FastSpeech2(**fastspeech2_config, n_speakers=n_speakers)
    fastspeech2.to(device)

    # model2: intensity extractor
    rank_model = RankModel(**rank_model_config, n_emotions=n_emotions, n_mels=n_mels)
    rank_model.load_state_dict(torch.load(rank_model_pth_path, map_location=device))
    intensity_extractor = rank_model.intensity_extractor
    intensity_extractor.to(device).eval().requires_grad_(False)


    # model3: hifigan vocoder
    vocoder = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz", savedir=vocoder_path)

    # loss
    criterion = Loss(**loss_config)
    criterion.to(device)

    # optimizer
    optimizer = torch.optim.AdamW(fastspeech2.parameters(), lr=lr)

    # -- tensorboard
    exp_path = os.path.join(exp_base_path, 'fastspeech2')
    exp_path = increment_path(exp_path)
    writer = SummaryWriter(exp_path)

    # -- stats
    global_step = 0
    patience = 0
    best_valid_loss = float('inf')
    
    for epoch in range(n_epochs):
        
        train_one_epoch(train_dataloader, fastspeech2, intensity_extractor, criterion, optimizer, device, epoch, exp_path, writer)
        val_loss = valid_one_epoch(valid_dataloader, fastspeech2, intensity_extractor, vocoder, criterion, device, epoch, exp_path, writer)

        if val_loss < best_valid_loss:
            patience = 0
            best_valid_loss = val_loss
            print(f'New best validation loss: {best_valid_loss:.4f}')
            torch.save(fastspeech2.state_dict(), os.path.join(exp_path, 'best_model.pth'))
        else:
            patience += 1
            print(f'Validation loss did not improve: {patience}/{max_patience}')
            if patience >= max_patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

        global_step += len(train_dataloader)
        if global_step >= max_iterations:
            print(f"Reached max iterations: {max_iterations}. Stopping training.")
            break

    writer.close()
    print("Training completed.")

if __name__ == '__main__':
    config = yaml.safe_load(open('parameter.yaml', 'r'))
    train(config)