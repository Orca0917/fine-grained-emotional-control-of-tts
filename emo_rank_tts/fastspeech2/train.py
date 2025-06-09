import yaml
import torch

from tqdm import tqdm
from loss import Loss
from model import FastSpeech2, FineGraindModel
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from speechbrain.inference.vocoders import HIFIGAN
from dataset import FastSpeech2Dataset, TextMelCollateWithAlignment
from util import batch_to_device, plot_fastspeech2_melspecs, increment_path, synthesize_sample



def train_one_epoch(dataloader, model, rank_model, criterion, optim, device, epoch, exp_path, writer):
    model.train()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}', unit='batch', dynamic_ncols=True)
    epoch_avg_loss = defaultdict(float)

    for idx, batch in enumerate(pbar):

        batch = batch_to_device(batch, device)
        (
            phoneme, speakers, phon_len, mel_target, target_pitch, target_energy,
            target_duration, mel_length, labels, wavs, rank_X, emotion
        ) = batch

        # intensity extraction
        with torch.no_grad():
            B = rank_X.size(0)
            lambdas = torch.ones((2, B), device=device)
            I = rank_model(rank_X, rank_X, emotion, mel_length, lambdas)[2]

        B, T = phoneme.shape
        _, max_mel_len, D = I.shape
        intensity_rep = torch.zeros((B, T, D), device=device)

        for b in range(B):


            real_T = phon_len[b].item()
            durations = target_duration[b].long()[:real_T]
            real_mel_len = int(durations.sum().item())

            I_b = I[b, :real_mel_len, :]

            phon_idx = torch.repeat_interleave(
                torch.arange(real_T, device=device), durations
            )

            sum_rep = torch.zeros((real_T, D), device=device)
            sum_rep.index_add_(0, phon_idx, I_b)

            denom = durations.unsqueeze(1).float().clamp(min=1.0)
            intensity_rep[b, :real_T, :] = sum_rep / denom

        
        # forward pass
        predictions = model(phoneme, speakers, target_duration, target_pitch, target_energy, intensity=intensity_rep)

        # compute loss
        targets = (mel_target, target_duration, target_pitch, target_energy, mel_length, phon_len)
        loss = criterion(predictions, targets, epoch)

        # backward pass
        optim.zero_grad()
        loss['total_loss'].backward()
        optim.step()

        # accumulate loss
        for loss_name, loss_value in loss.items():
            epoch_avg_loss[loss_name] += loss_value

        if idx == 0 and epoch % 10 == 0:
            melspecs = predictions[0].cpu().detach().numpy()
            y_melspecs = mel_target.cpu().detach().numpy()
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
                phoneme, speakers, phon_len, mel_target, target_pitch, target_energy,
                target_duration, mel_length, labels, wavs, rank_X, emotion
            ) = batch

            # intensity extraction
            B = rank_X.size(0)
            lambdas = torch.ones((2, B), device=device)
            I = rank_model(rank_X, rank_X, emotion, mel_length, lambdas)[2]

            B, T = phoneme.shape
            _, max_mel_len, D = I.shape
            intensity_rep = torch.zeros((B, T, D), device=device)

            for b in range(B):
                real_T = phon_len[b].item()
                durations = target_duration[b].long()[:real_T]
                real_mel_len = int(durations.sum().item())

                I_b = I[b, :real_mel_len, :]

                phon_idx = torch.repeat_interleave(
                    torch.arange(real_T, device=device), durations
                )

                sum_rep = torch.zeros((real_T, D), device=device)
                sum_rep.index_add_(0, phon_idx, I_b)

                denom = durations.unsqueeze(1).float().clamp(min=1.0)
                intensity_rep[b, :real_T, :] = sum_rep / denom

            # forward pass
            predictions = model(phoneme, speakers, target_duration,
                                target_pitch, target_energy,
                                intensity=intensity_rep)

            # compute loss
            targets = (mel_target, target_duration,
                       target_pitch, target_energy,
                       mel_length, phon_len)
            loss = criterion(predictions, targets)

            # accumulate loss
            for loss_name, loss_value in loss.items():
                epoch_avg_loss[loss_name] += loss_value

            if idx == 0 and epoch % 10 == 0:
                melspecs = predictions[0]
                y_melspecs = mel_target
                synthesize_sample(vocoder, melspecs, y_melspecs, exp_path, epoch)
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


def train(config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    preprocessed_path       = config['path']['preprocessed_path']
    experiment_path         = config['path']['experiment_path']
    vocoder_path            = config['path']['vocoder_path']
    noise_symbol            = config['preprocessing']['noise_symbol']
    speakers                = config['preprocessing']['speakers']
    emotions                = config['preprocessing']['emotions']
    lr                      = config['train']['learning_rate']
    batch_size              = config['train']['batch_size']
    n_epochs                = config['train']['n_epochs']
    fastspeech2_config      = config['model']['fastspeech2']
    rank_model_config       = config['model']['rank_model']
    n_mels                  = config['audio']['n_mels']
    rank_model_path_path    = config['inference']['best_pth_path']
    loss_config             = config['loss']
    

    # Load dataset
    train_dataset = FastSpeech2Dataset(preprocessed_path, noise_symbol, 
                                 speakers, emotions, mode='train')
    valid_dataset = FastSpeech2Dataset(preprocessed_path, noise_symbol, 
                                 speakers, emotions, mode='valid')
    
    # dataloaoder
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=TextMelCollateWithAlignment(),
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=TextMelCollateWithAlignment(),
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    # model
    model = FastSpeech2(**fastspeech2_config, n_speakers=len(speakers))
    model.to(device)

    # model2: intensity extractor
    rank_model = FineGraindModel(**rank_model_config, 
                                 n_speakers=len(speakers), 
                                 n_emotions=len(emotions),
                                 n_mels=n_mels)
    rank_model.load_state_dict(torch.load(rank_model_path_path))
    rank_model.to(device)
    rank_model.eval()

    # model3: hifigan vocoder
    vocoder = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz", 
                                   savedir=vocoder_path)

    # freeze rank model
    for param in rank_model.parameters():
        param.requires_grad = False


    # loss
    criterion = Loss(**loss_config)
    criterion.to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -- tensorboard
    exp_path = increment_path(experiment_path)
    writer = SummaryWriter(exp_path)


    # -- stats
    global_step = 0
    best_valid_loss = float('inf')
    
    for epoch in range(n_epochs):
        
        train_one_epoch(train_dataloader, model, rank_model, criterion, optimizer, device,
                        epoch, exp_path, writer)
        
        # TODO: validation step, synthesize with vocoder
        valid_one_epoch(valid_dataloader, model, rank_model, vocoder, criterion, device, epoch, exp_path, writer)

        # TODO: save best model


        global_step += len(train_dataloader)

    pass

if __name__ == '__main__':
    config = yaml.safe_load(open('parameter.yaml', 'r'))
    train(config)