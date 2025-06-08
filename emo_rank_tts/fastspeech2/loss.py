import torch
import torch.nn as nn
from speechbrain.lobes.models.FastSpeech2 import SSIMLoss


class Loss(nn.Module):
    """Loss Computation

    Arguments
    ---------
    log_scale_durations: bool
        applies logarithm to target durations
    ssim_loss_weight: float
        weight for ssim loss
    duration_loss_weight: float
        weight for the duration loss
    pitch_loss_weight: float
        weight for the pitch loss
    energy_loss_weight: float
        weight for the energy loss
    mel_loss_weight: float
        weight for the mel loss
    postnet_mel_loss_weight: float
        weight for the postnet mel loss
    spn_loss_weight: float
        weight for spn loss
    spn_loss_max_epochs: int
        Max number of epochs
    """

    def __init__(
        self,
        log_scale_durations,
        ssim_loss_weight,
        duration_loss_weight,
        pitch_loss_weight,
        energy_loss_weight,
        mel_loss_weight,
        postnet_mel_loss_weight,
        spn_loss_weight=1.0,
        spn_loss_max_epochs=8,
    ):
        super().__init__()

        self.ssim_loss = SSIMLoss()
        self.mel_loss = nn.MSELoss()
        self.postnet_mel_loss = nn.MSELoss()
        self.dur_loss = nn.MSELoss()
        self.pitch_loss = nn.MSELoss()
        self.energy_loss = nn.MSELoss()
        self.log_scale_durations = log_scale_durations
        self.ssim_loss_weight = ssim_loss_weight
        self.mel_loss_weight = mel_loss_weight
        self.postnet_mel_loss_weight = postnet_mel_loss_weight
        self.duration_loss_weight = duration_loss_weight
        self.pitch_loss_weight = pitch_loss_weight
        self.energy_loss_weight = energy_loss_weight
        self.spn_loss_weight = spn_loss_weight
        self.spn_loss_max_epochs = spn_loss_max_epochs


    def forward(self, predictions, targets, current_epoch):
        """Computes the value of the loss function and updates stats

        Arguments
        ---------
        predictions: tuple
            model predictions
        targets: tuple
            ground truth data
        current_epoch: int
            The count of the current epoch.

        Returns
        -------
        loss: torch.Tensor
            the loss value
        """
        (
            mel_target,
            target_durations,
            target_pitch,
            target_energy,
            mel_length,
            phon_len,
            # spn_labels,
        ) = targets
        assert len(mel_target.shape) == 3
        (
            mel_out,
            postnet_mel_out,
            log_durations,
            predicted_pitch,
            average_pitch,
            predicted_energy,
            average_energy,
            mel_lens,
            # spn_preds,
        ) = predictions

        predicted_pitch = predicted_pitch.squeeze(-1)
        predicted_energy = predicted_energy.squeeze(-1)

        target_pitch = average_pitch.squeeze(-1)
        target_energy = average_energy.squeeze(-1)

        log_durations = log_durations.squeeze(-1)
        if self.log_scale_durations:
            log_target_durations = torch.log1p(target_durations.float())
        # change this to perform batch level using padding mask

        for i in range(mel_target.shape[0]):
            if i == 0:
                mel_loss = self.mel_loss(
                    mel_out[i, : mel_length[i], :],
                    mel_target[i, : mel_length[i], :],
                )
                postnet_mel_loss = self.postnet_mel_loss(
                    postnet_mel_out[i, : mel_length[i], :],
                    mel_target[i, : mel_length[i], :],
                )
                dur_loss = self.dur_loss(
                    log_durations[i, : phon_len[i]],
                    log_target_durations[i, : phon_len[i]].to(torch.float32),
                )
                pitch_loss = self.pitch_loss(
                    predicted_pitch[i, : mel_length[i]],
                    target_pitch[i, : mel_length[i]].to(torch.float32),
                )
                energy_loss = self.energy_loss(
                    predicted_energy[i, : mel_length[i]],
                    target_energy[i, : mel_length[i]].to(torch.float32),
                )
            else:
                mel_loss = mel_loss + self.mel_loss(
                    mel_out[i, : mel_length[i], :],
                    mel_target[i, : mel_length[i], :],
                )
                postnet_mel_loss = postnet_mel_loss + self.postnet_mel_loss(
                    postnet_mel_out[i, : mel_length[i], :],
                    mel_target[i, : mel_length[i], :],
                )
                dur_loss = dur_loss + self.dur_loss(
                    log_durations[i, : phon_len[i]],
                    log_target_durations[i, : phon_len[i]].to(torch.float32),
                )
                pitch_loss = pitch_loss + self.pitch_loss(
                    predicted_pitch[i, : mel_length[i]],
                    target_pitch[i, : mel_length[i]].to(torch.float32),
                )
                energy_loss = energy_loss + self.energy_loss(
                    predicted_energy[i, : mel_length[i]],
                    target_energy[i, : mel_length[i]].to(torch.float32),
                )
        ssim_loss = self.ssim_loss(mel_out, mel_target, mel_length)
        mel_loss = torch.div(mel_loss, len(mel_target))
        postnet_mel_loss = torch.div(postnet_mel_loss, len(mel_target))
        dur_loss = torch.div(dur_loss, len(mel_target))
        pitch_loss = torch.div(pitch_loss, len(mel_target))
        energy_loss = torch.div(energy_loss, len(mel_target))

        # spn_loss = bce_loss(spn_preds, spn_labels)
        # if current_epoch > self.spn_loss_max_epochs:
        #     self.spn_loss_weight = 0

        total_loss = (
            ssim_loss * self.ssim_loss_weight
            + mel_loss * self.mel_loss_weight
            + postnet_mel_loss * self.postnet_mel_loss_weight
            + dur_loss * self.duration_loss_weight
            + pitch_loss * self.pitch_loss_weight
            + energy_loss * self.energy_loss_weight
            # + spn_loss * self.spn_loss_weight
        )

        loss = {
            "total_loss": total_loss,
            "ssim_loss": ssim_loss * self.ssim_loss_weight,
            "mel_loss": mel_loss * self.mel_loss_weight,
            "postnet_mel_loss": postnet_mel_loss * self.postnet_mel_loss_weight,
            "dur_loss": dur_loss * self.duration_loss_weight,
            "pitch_loss": pitch_loss * self.pitch_loss_weight,
            "energy_loss": energy_loss * self.energy_loss_weight,
            # "spn_loss": spn_loss * self.spn_loss_weight,
        }
        return loss