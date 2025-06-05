import yaml
import torch

from tqdm import tqdm
from loss import Loss
from model import FastSpeech2
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from dataset import FastSpeech2Dataset, TextMelCollateWithAlignment
from util import batch_to_device, plot_fastspeech2_melspecs, increment_path



def train_one_epoch(dataloader, model, criterion, optim, device, epoch, exp_path, writer):
    model.train()
    
    epoch_avg_loss = defaultdict(float)

    for idx, batch in enumerate(tqdm(dataloader)):
        batch = batch_to_device(batch, device)

        phoneme, speakers, phon_len, mel_target, target_pitch, target_energy, target_duration, mel_length, labels, wavs = batch

        # Forward pass
        predictions = model(phoneme, speakers, target_duration, target_pitch, target_energy)

        # Compute loss
        targets = (mel_target, target_duration, target_pitch, target_energy, mel_length, phon_len)
        loss = criterion(predictions, targets, epoch)

        optim.zero_grad()
        loss['total_loss'].backward()
        optim.step()

        # Accumulate loss
        for loss_name, loss_value in loss.items():
            epoch_avg_loss[loss_name] += loss_value

        if idx == 0 and epoch % 10 == 0:
            melspecs = predictions[0].cpu().detach().numpy()
            y_melspecs = mel_target.cpu().detach().numpy()
            plot_fastspeech2_melspecs(melspecs, y_melspecs, epoch, exp_path)

    
    epoch_avg_loss = {k: v / len(dataloader) for k, v in epoch_avg_loss.items()}
    for k, v in epoch_avg_loss.items():
        writer.add_scalar(f'Loss/{k}', v, epoch)
    
    return epoch_avg_loss['total_loss']



def train(config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    preprocessed_path = config['path']['preprocessed_path']
    experiment_path = config['path']['experiment_path']
    noise_symbol = config['preprocessing']['noise_symbol']
    speakers = config['preprocessing']['speakers']
    emotions = config['preprocessing']['emotions']
    batch_size = config['train']['batch_size']
    fastspeech2_config = config['model']['fastspeech2']
    loss_config = config['loss']
    lr = config['train']['learning_rate']
    n_epochs = config['train']['n_epochs']

    # Load dataset
    dataset = FastSpeech2Dataset(preprocessed_path, noise_symbol, 
                                 speakers, emotions, mode='train')
    
    # dataloaoder
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=TextMelCollateWithAlignment(),
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    # model
    model = FastSpeech2(**fastspeech2_config, n_speakers=len(speakers))
    model.to(device)

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
        
        train_one_epoch(dataloader, model, criterion, optimizer, device,
                        epoch, exp_path, writer)
        
        global_step += len(dataloader)

    pass

if __name__ == '__main__':
    config = yaml.safe_load(open('parameter.yaml', 'r'))
    train(config)