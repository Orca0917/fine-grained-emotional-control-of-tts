import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt



# plot only one melspectrogram
def plot_melspectrogram(mel, save_path=None):

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(mel, aspect='auto', origin='lower', interpolation='none', cmap='viridis')
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('Mel bands')
    ax.set_title('Melspectrogram')

    cbar = fig.colorbar(im, ax=ax, format='%+2.0f dB')
    cbar.set_label('Amplitude (dB)')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    

# plot melspectrograms for GT and predicted
def plot_melspectrograms(mel_gt, mel_pred, save_path=None):
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    axes[0].imshow(mel_gt, aspect='auto', origin='lower', interpolation='none', cmap='viridis')
    axes[0].set_title('Ground Truth Melspectrogram')
    axes[0].set_xlabel('Time (frames)')
    axes[0].set_ylabel('Mel bands')

    axes[1].imshow(mel_pred, aspect='auto', origin='lower', interpolation='none', cmap='viridis')
    axes[1].set_title('Predicted Melspectrogram')
    axes[1].set_xlabel('Time (frames)')
    axes[1].set_ylabel('Mel bands')

    cbar = fig.colorbar(axes[0].images[0], ax=axes, format='%+2.0f dB')
    cbar.set_label('Amplitude (dB)')

    fig.subplots_adjust(hspace=0.6, right=0.75)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

# plot all features: mel, pitch, energy
def plot_features(mel, pitch, energy, stat, save_path=None):

    p_min, p_max, p_mean, p_std = stat['pitch']
    e_min, e_max, e_mean, e_std = stat['energy']
    
    p_min = p_min * p_std + p_mean
    p_max = p_max * p_std + p_mean

    e_min = e_min * e_std + e_mean
    e_max = e_max * e_std + e_mean

    pitch = pitch * p_std + p_mean
    energy = energy * e_std + e_mean

    @staticmethod
    def add_axis(fig, ax):
        ax1 = fig.add_axes(ax.get_position(), anchor='W')
        ax1.set_facecolor('none')
        return ax1

    
    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(mel, aspect='auto', origin='lower', interpolation='none', cmap='viridis')
    ax.set_anchor('W')
    ax.tick_params(labelsize='x-small', left=False, labelleft=False, right=False, labelright=False)
    ax.set_xlabel('Time (frames)')
    ax.set_title('Melspectrogram')

    ax1 = add_axis(fig, ax)
    ax1.set_xlim(0, mel.shape[1])
    ax1.plot(pitch, color='tomato', label='Pitch', linewidth=2)
    ax1.set_ylim(p_min, p_max)
    ax1.set_ylabel('Pitch (Hz)', color='tomato')
    ax1.tick_params(labelsize='x-small', colors='tomato', bottom=False, labelbottom=False)
    
    ax2 = add_axis(fig, ax)
    ax2.set_xlim(0, mel.shape[1])
    ax2.plot(energy, color='darkviolet', label='Energy', linewidth=2)
    ax2.set_ylim(e_min, e_max)
    ax2.set_ylabel('Energy (dB)', color='darkviolet')
    ax2.yaxis.set_label_position("right")
    ax2.tick_params(labelsize='x-small', colors='darkviolet', bottom=False, labelbottom=False, 
                    left=False, labelleft=False, right=True, labelright=True)

    # plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def increment_path(base_path):
     
    exp_num = 1
    while True:
        path = os.path.join(base_path, f'exp_{exp_num}')
        if not os.path.exists(path):
            os.makedirs(path)
            return path
        exp_num += 1