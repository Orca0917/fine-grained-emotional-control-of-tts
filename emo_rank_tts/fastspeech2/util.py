import os
import numpy as np
import matplotlib.pyplot as plt
import speechbrain


SIL_PHONES = ['sil', 'spn', 'sp', '']
VALID_TOKENS = ['@'] + speechbrain.utils.text_to_sequence.valid_symbols + SIL_PHONES


def phoneme2sequence(phoneme):
    seq = [VALID_TOKENS.index(token) for token in phoneme]
    return seq


def sequence2phoneme(sequence):
    phoneme = [VALID_TOKENS[i] for i in sequence]
    return phoneme


def batch_to_device(batch, device):
    return (
        batch[0].to(device),  # phoneme
        batch[1].to(device),  # speaker
        batch[2].to(device),  # input_lengths
        batch[3].to(device),  # mel
        batch[4].to(device),  # pitch
        batch[5].to(device),  # energy
        batch[6].to(device),  # duration
        batch[7].to(device),  # output_lengths
        batch[8],             # labels (strings)
        batch[9],             # wavs (file paths)
        batch[10].to(device), # rank_X
        batch[11].to(device), # emotion
    )


def plot_fastspeech2_melspecs(melspecs, y_melspecs, epoch, exp_path, train=True):

    melspecs = melspecs[:8, :, :]
    y_melsspecs = y_melspecs[:8, :, :]

    all_melspecs = np.concatenate((melspecs, y_melspecs), axis=0)
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 10))
    for ax_idx, (ax, mel) in enumerate(zip(axes.flatten(), all_melspecs)):
        ax.imshow(mel.T, aspect='auto', origin='lower', interpolation='none')
        if ax_idx < len(melspecs):
            label = f"Pred {ax_idx + 1}"
            color = 'blue'
        else:
            label = f"GT {ax_idx - len(melspecs) + 1}"
            color = 'red'

        ax.text(
            0.95, 0.95, label,
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax.transAxes,
            fontsize=12,
            fontweight='bold',
            color=color,
        )
    
    plt.tight_layout()
    save_path = os.path.join(exp_path, f"epoch_{epoch}.png" if train else f"valid_epoch_{epoch}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


def increment_path(base_path):
     
    exp_num = 1
    while True:
        path = os.path.join(base_path, f'exp_{exp_num}')
        if not os.path.exists(path):
            os.makedirs(path)
            return path
        exp_num += 1


def synthesize_sample(vocoder, melspecs, y_melspecs, exp_path, epoch):
    melspecs = melspecs[:8, :, :]
    y_melspecs = y_melspecs[:8, :, :]

    for i, (mel, y_mel) in enumerate(zip(melspecs, y_melspecs)):
        mel = mel.unsqueeze(0)
        y_mel = y_mel.unsqueeze(0)

        wav = vocoder.decode_batch(mel)
        y_wav = vocoder.decode_batch(y_mel)

        wav_path = os.path.join(exp_path, f'epoch_{epoch}_sample_{i + 1}_pred.wav')
        y_wav_path = os.path.join(exp_path, f'epoch_{epoch}_sample_{i + 1}_gt.wav')

        speechbrain.utils.dataio.write_audio(wav_path, wav.squeeze(0), 22050)
        speechbrain.utils.dataio.write_audio(y_wav_path, y_wav.squeeze(0), 22050)