import os
import yaml
import torch
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from tqdm import tqdm
from loss import RankLoss
from model import FineGraindModel
from matplotlib.lines import Line2D
from util import set_seed, increment_path
from torch.utils.tensorboard import SummaryWriter
from dataset import FineGrainedDataset, collate_fn


def train_one_epoch(dataloader, model, criterion, epoch, optim, writer, device):
    
    model.train()

    pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}', unit='batch')
    losses, mixup_losses, rank_losses = [], [], []

    for batch in pbar:

        emo_X = batch['emo_X'].to(device)
        neu_X = batch['neu_X'].to(device)
        speakers = batch['speakers'].to(device)
        emotions = batch['emotions'].to(device)
        length = batch['length'].to(device)
        targets = (emotions, torch.full_like(emotions, 0, device=device))

        # forward pass
        predictions = model(emo_X, neu_X, emotions, length)

        # compute loss
        loss, L_mixup, L_rank = criterion(predictions, targets)

        # backward pass
        optim.zero_grad()
        loss.backward()
        optim.step()

        loss, mixup_loss, rank_loss = loss.item(), L_mixup.item(), L_rank.item()
        losses.append(loss)
        mixup_losses.append(mixup_loss)
        rank_losses.append(rank_loss)

        pbar.set_postfix(loss="{:.04f}".format(loss),
                         mixup_loss="{:.04f}".format(mixup_loss),
                         rank_loss="{:.04f}".format(rank_loss))

    pbar.close()

    avg_loss = np.mean(losses)
    avg_mixup_loss = np.mean(mixup_losses)
    avg_rank_loss = np.mean(rank_losses)

    
    # tensorboard logging
    writer.add_scalar('train/loss', avg_loss, epoch)
    writer.add_scalar('train/mixup_loss', avg_mixup_loss, epoch)
    writer.add_scalar('train/rank_loss', avg_rank_loss, epoch)


    best_avg_loss = avg_loss.min()
    return best_avg_loss


def validate_one_epoch(dataloader, model, criterion, epoch, writer, device,
                       exp_path, emotions, speakers, colors, markers,):
    
    model.eval()
    n_emotions = len(emotions)
    n_speakers = len(speakers)

    pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}', unit='batch')
    losses, mixup_losses, rank_losses = [], [], []
    h_list, label_list, speaker_list, lam_list = [], [], [], []

    for batch in pbar:

        emo_X = batch['emo_X'].to(device)
        neu_X = batch['neu_X'].to(device)
        speaker = batch['speakers'].to(device)
        emotion = batch['emotions'].to(device)
        length = batch['length'].to(device)
        targets = (emotion, torch.full_like(emotion, 0, device=device))

        B = emo_X.size(0)
        lambdas = torch.linspace(0, 1, steps=B).unsqueeze(0).repeat(2,1).to(device)

        # forward pass
        predictions = model(emo_X, neu_X, emotion, length, lambdas=lambdas)

        # compute loss
        loss, L_mixup, L_rank = criterion(predictions, targets)

        loss, mixup_loss, rank_loss = loss.item(), L_mixup.item(), L_rank.item()
        losses.append(loss)
        mixup_losses.append(mixup_loss)
        rank_losses.append(rank_loss)

        h_list.append(predictions[2].detach().cpu().numpy())
        label_list.append(emotion.detach().cpu().numpy())
        speaker_list.append(speaker.detach().cpu().numpy())
        lam_list.append(lambdas[0, :].detach().cpu().numpy())

        pbar.set_postfix(loss="{:.04f}".format(loss),
                         mixup_loss="{:.04f}".format(mixup_loss),
                         rank_loss="{:.04f}".format(rank_loss))

    pbar.close()

    # compute stats
    avg_loss = np.mean(losses)
    avg_mixup_loss = np.mean(mixup_losses)
    avg_rank_loss = np.mean(rank_losses)

    
    # tensorboard logging
    writer.add_scalar('valid/loss', avg_loss, epoch)
    writer.add_scalar('valid/mixup_loss', avg_mixup_loss, epoch)
    writer.add_scalar('valid/rank_loss', avg_rank_loss, epoch)

    

    # visualization
    h_array = np.vstack(h_list)
    labels = np.concatenate(label_list)
    speaker = np.concatenate(speaker_list)
    lambdas = np.concatenate(lam_list)
    alphas = np.minimum(1.0, lambdas + 0.1)

    pca = sklearn.decomposition.PCA(n_components=50).fit_transform(h_array)
    tsne = sklearn.manifold.TSNE(n_components=2, init='pca').fit_transform(pca)

    fig, ax = plt.subplots(figsize=(10, 10))
    for cls in range(n_emotions):
        for spk in range(n_speakers):
            mask = (labels == cls) & (speaker == spk)
            if not np.any(mask):
                continue
            ax.scatter(
                tsne[mask, 0], tsne[mask, 1],
                c=colors[cls], marker=markers[spk],
                s=15, alpha=alphas[mask], label=None,
            )


    emotion_handles = [
        mpatches.Patch(color=colors[i], label=emotions[i])
        for i in range(n_emotions)
    ]
    speaker_handles = [
        Line2D([], [], color='black', marker=markers[i], linestyle='None', label=speakers[i])
        for i in range(n_speakers)
    ]

    legend1 = ax.legend(handles=emotion_handles, title='Emotion', loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.add_artist(legend1)
    ax.legend(handles=speaker_handles, title='Speaker', loc='upper left', bbox_to_anchor=(1.05, 0.6))
    
    ax.set_title("t-SNE of hidden features by emotion")
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(exp_path, f'tsne_epoch_{epoch}.png'))
    plt.close()

    return avg_loss



def train(config):

    # -- parameters
    preprocessed_path   = config['path']['preprocessed_path']
    experiment_path     = config['path']['experiment_path']
    speakers            = config['preprocessing']['speakers']
    emotions            = config['preprocessing']['emotions']
    n_mels              = config['audio']['n_mels']
    n_heads             = config['model']['n_heads']
    n_encoder_layers    = config['model']['n_encoder_layers']
    hidden_dim          = config['model']['hidden_dim']
    kernel_size         = config['model']['kernel_size']
    dropout             = config['model']['dropout']
    alpha               = config['model']['alpha']
    beta                = config['model']['beta']
    batch_size          = config['train']['batch_size']
    lr                  = config['train']['learning_rate']
    max_iterations      = config['train']['max_iterations']
    colors              = config['misc']['colors']
    markers             = config['misc']['markers']
    n_speakers          = len(speakers)
    n_emotions          = len(emotions)

    set_seed(42)

    # -- dataset
    train_dataset = FineGrainedDataset(preprocessed_path, speakers, emotions, split='train')
    valid_dataset = FineGrainedDataset(preprocessed_path, speakers, emotions, split='test')


    # -- dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )


    # -- model
    model = FineGraindModel(n_mels, n_heads, n_speakers, n_emotions,
                      n_encoder_layers, hidden_dim, kernel_size, dropout)
    model = model.to(device)


    # -- loss
    criterion = RankLoss(alpha, beta)
    criterion = criterion.to(device)


    # -- optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)


    # -- tensorboard
    exp_path = increment_path(experiment_path)
    writer = SummaryWriter(exp_path)


    # -- stats
    global_step = 0
    best_valid_loss = float('inf')


    # -- training loop
    validate_one_epoch(valid_dataloader, model, criterion, 0, writer, 
                       device, exp_path, emotions, speakers, colors, markers)
    for epoch in range(config['train']['n_epohcs']):
        train_one_epoch(train_dataloader, model, criterion, epoch, optim, writer, device)
        val_loss = validate_one_epoch(valid_dataloader, model, criterion, epoch, writer,
                           device, exp_path, emotions, speakers, colors, markers)
        
        # save best model
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            print(f'New best validation loss: {best_valid_loss:.4f}')
            torch.save(model.state_dict(), os.path.join(exp_path, 'best_model.pth'))
        
        
        global_step += len(train_dataloader)
        if global_step >= max_iterations:
            print("Reached maximum iterations, stopping training.")
            break


if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Load configuration
    config = yaml.safe_load(open('parameter.yaml'))

    train(config)