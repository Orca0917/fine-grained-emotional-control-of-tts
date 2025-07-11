"""
Neural network modules for the FastSpeech 2: Fast and High-Quality End-to-End Text to Speech
synthesis model
Authors
* Sathvik Udupa 2022
* Pradnya Kandarkar 2023
* Yingzhi Wang 2023
"""

import torch
from torch import nn

from speechbrain.lobes.models.transformer.Transformer import (
    PositionalEncoding,
    TransformerEncoder,
    get_key_padding_mask,
    get_mask_from_lengths,
)
from speechbrain.nnet import CNN, linear
from speechbrain.nnet.embedding import Embedding
from speechbrain.lobes.models.FastSpeech2 import (
    EncoderPreNet,
    DurationPredictor,
    PostNet,
    upsample,
    average_over_durations,
)




class FastSpeech2(nn.Module):
    """The FastSpeech2 text-to-speech model.
    This class is the main entry point for the model, which is responsible
    for instantiating all submodules, which, in turn, manage the individual
    neural network layers
    Simplified STRUCTURE: input->token embedding ->encoder ->duration/pitch/energy predictor ->duration
    upsampler -> decoder -> output
    During training, teacher forcing is used (ground truth durations are used for upsampling)

    Arguments
    ---------
    enc_num_layers: int
        number of transformer layers (TransformerEncoderLayer) in encoder
    enc_num_head: int
        number of multi-head-attention (MHA) heads in encoder transformer layers
    enc_d_model: int
        the number of expected features in the encoder
    enc_ffn_dim: int
        the dimension of the feedforward network model
    enc_k_dim: int
        the dimension of the key
    enc_v_dim: int
        the dimension of the value
    enc_dropout: float
        Dropout for the encoder
    dec_num_layers: int
        number of transformer layers (TransformerEncoderLayer) in decoder
    dec_num_head: int
        number of multi-head-attention (MHA) heads in decoder transformer layers
    dec_d_model: int
        the number of expected features in the decoder
    dec_ffn_dim: int
        the dimension of the feedforward network model
    dec_k_dim: int
        the dimension of the key
    dec_v_dim: int
        the dimension of the value
    dec_dropout: float
        dropout for the decoder
    normalize_before: bool
        whether normalization should be applied before or after MHA or FFN in Transformer layers.
    ffn_type: str
        whether to use convolutional layers instead of feed forward network inside transformer layer.
    ffn_cnn_kernel_size_list: list of int
        conv kernel size of 2 1d-convs if ffn_type is 1dcnn
    n_char: int
        the number of symbols for the token embedding
    n_mels: int
        number of bins in mel spectrogram
    postnet_embedding_dim: int
       output feature dimension for convolution layers
    postnet_kernel_size: int
       postnet convolution kernel size
    postnet_n_convolutions: int
       number of convolution layers
    postnet_dropout: float
        dropout probability for postnet
    padding_idx: int
        the index for padding
    dur_pred_kernel_size: int
        the convolution kernel size in duration predictor
    pitch_pred_kernel_size: int
        kernel size for pitch prediction.
    energy_pred_kernel_size: int
        kernel size for energy prediction.
    variance_predictor_dropout: float
        dropout probability for variance predictor (duration/pitch/energy)

    Example
    -------
    >>> import torch
    >>> from speechbrain.lobes.models.FastSpeech2 import FastSpeech2
    >>> model = FastSpeech2(
    ...    enc_num_layers=6,
    ...    enc_num_head=2,
    ...    enc_d_model=384,
    ...    enc_ffn_dim=1536,
    ...    enc_k_dim=384,
    ...    enc_v_dim=384,
    ...    enc_dropout=0.1,
    ...    dec_num_layers=6,
    ...    dec_num_head=2,
    ...    dec_d_model=384,
    ...    dec_ffn_dim=1536,
    ...    dec_k_dim=384,
    ...    dec_v_dim=384,
    ...    dec_dropout=0.1,
    ...    normalize_before=False,
    ...    ffn_type='1dcnn',
    ...    ffn_cnn_kernel_size_list=[9, 1],
    ...    n_char=40,
    ...    n_mels=80,
    ...    postnet_embedding_dim=512,
    ...    postnet_kernel_size=5,
    ...    postnet_n_convolutions=5,
    ...    postnet_dropout=0.5,
    ...    padding_idx=0,
    ...    dur_pred_kernel_size=3,
    ...    pitch_pred_kernel_size=3,
    ...    energy_pred_kernel_size=3,
    ...    variance_predictor_dropout=0.5)
    >>> inputs = torch.tensor([
    ...     [13, 12, 31, 14, 19],
    ...     [31, 16, 30, 31, 0],
    ... ])
    >>> input_lengths = torch.tensor([5, 4])
    >>> durations = torch.tensor([
    ...     [2, 4, 1, 5, 3],
    ...     [1, 2, 4, 3, 0],
    ... ])
    >>> mel_post, postnet_output, predict_durations, predict_pitch, avg_pitch, predict_energy, avg_energy, mel_lens = model(inputs, durations=durations)
    >>> mel_post.shape, predict_durations.shape
    (torch.Size([2, 15, 80]), torch.Size([2, 5]))
    >>> predict_pitch.shape, predict_energy.shape
    (torch.Size([2, 5, 1]), torch.Size([2, 5, 1]))
    """

    def __init__(
        self,
        # encoder parameters
        enc_num_layers,
        enc_num_head,
        enc_d_model,
        enc_ffn_dim,
        enc_k_dim,
        enc_v_dim,
        enc_dropout,
        # decoder parameters
        dec_num_layers,
        dec_num_head,
        dec_d_model,
        dec_ffn_dim,
        dec_k_dim,
        dec_v_dim,
        dec_dropout,
        normalize_before,
        ffn_type,
        ffn_cnn_kernel_size_list,
        n_char,
        n_mels,
        postnet_embedding_dim,
        postnet_kernel_size,
        postnet_n_convolutions,
        postnet_dropout,
        padding_idx,
        dur_pred_kernel_size,
        pitch_pred_kernel_size,
        energy_pred_kernel_size,
        variance_predictor_dropout,
        n_speakers,
    ):
        super().__init__()
        self.enc_num_head = enc_num_head
        self.dec_num_head = dec_num_head
        self.padding_idx = padding_idx
        self.sinusoidal_positional_embed_encoder = PositionalEncoding(
            enc_d_model
        )
        self.sinusoidal_positional_embed_decoder = PositionalEncoding(
            dec_d_model
        )

        self.speaker_emb = Embedding(
            num_embeddings=n_speakers,
            embedding_dim=enc_d_model,
            # padding_idx=padding_idx,
        )
        self.concat_proj = linear.Linear(
            n_neurons=enc_d_model,
            input_size=enc_d_model + enc_d_model + 5,  # token embedding + speaker embedding + intensity
            bias=False,
        )

        self.encPreNet = EncoderPreNet(
            n_char, padding_idx, out_channels=enc_d_model
        )
        self.durPred = DurationPredictor(
            in_channels=enc_d_model,
            out_channels=enc_d_model,
            kernel_size=dur_pred_kernel_size,
            dropout=variance_predictor_dropout,
        )
        self.pitchPred = DurationPredictor(
            in_channels=enc_d_model,
            out_channels=enc_d_model,
            kernel_size=dur_pred_kernel_size,
            dropout=variance_predictor_dropout,
        )
        self.energyPred = DurationPredictor(
            in_channels=enc_d_model,
            out_channels=enc_d_model,
            kernel_size=dur_pred_kernel_size,
            dropout=variance_predictor_dropout,
        )
        self.pitchEmbed = CNN.Conv1d(
            in_channels=1,
            out_channels=enc_d_model,
            kernel_size=pitch_pred_kernel_size,
            padding="same",
            skip_transpose=True,
        )

        self.energyEmbed = CNN.Conv1d(
            in_channels=1,
            out_channels=enc_d_model,
            kernel_size=energy_pred_kernel_size,
            padding="same",
            skip_transpose=True,
        )
        self.encoder = TransformerEncoder(
            num_layers=enc_num_layers,
            nhead=enc_num_head,
            d_ffn=enc_ffn_dim,
            d_model=enc_d_model,
            kdim=enc_k_dim,
            vdim=enc_v_dim,
            dropout=enc_dropout,
            activation=nn.ReLU,
            normalize_before=normalize_before,
            ffn_type=ffn_type,
            ffn_cnn_kernel_size_list=ffn_cnn_kernel_size_list,
        )

        self.decoder = TransformerEncoder(
            num_layers=dec_num_layers,
            nhead=dec_num_head,
            d_ffn=dec_ffn_dim,
            d_model=dec_d_model,
            kdim=dec_k_dim,
            vdim=dec_v_dim,
            dropout=dec_dropout,
            activation=nn.ReLU,
            normalize_before=normalize_before,
            ffn_type=ffn_type,
            ffn_cnn_kernel_size_list=ffn_cnn_kernel_size_list,
        )

        self.linear = linear.Linear(n_neurons=n_mels, input_size=dec_d_model)
        self.postnet = PostNet(
            n_mel_channels=n_mels,
            postnet_embedding_dim=postnet_embedding_dim,
            postnet_kernel_size=postnet_kernel_size,
            postnet_n_convolutions=postnet_n_convolutions,
            postnet_dropout=postnet_dropout,
        )


    def forward(
        self,
        tokens,
        speakers,
        durations=None,
        pitch=None,
        energy=None,
        pace=1.0,
        pitch_rate=1.0,
        energy_rate=1.0,
        intensity=None,
    ):
        """forward pass for training and inference

        Arguments
        ---------
        tokens: torch.Tensor
            batch of input tokens
        durations: torch.Tensor
            batch of durations for each token. If it is None, the model will infer on predicted durations
        pitch: torch.Tensor
            batch of pitch for each frame. If it is None, the model will infer on predicted pitches
        energy: torch.Tensor
            batch of energy for each frame. If it is None, the model will infer on predicted energies
        pace: float
            scaling factor for durations
        pitch_rate: float
            scaling factor for pitches
        energy_rate: float
            scaling factor for energies

        Returns
        -------
        mel_post: torch.Tensor
            mel outputs from the decoder
        postnet_output: torch.Tensor
            mel outputs from the postnet
        predict_durations: torch.Tensor
            predicted durations of each token
        predict_pitch: torch.Tensor
            predicted pitches of each token
        avg_pitch: torch.Tensor
            target pitches for each token if input pitch is not None
            None if input pitch is None
        predict_energy: torch.Tensor
            predicted energies of each token
        avg_energy: torch.Tensor
            target energies for each token if input energy is not None
            None if input energy is None
        mel_length:
            predicted lengths of mel spectrograms
        """
        srcmask = get_key_padding_mask(tokens, pad_idx=self.padding_idx)  # (B, T)
        srcmask_inverted = (~srcmask).unsqueeze(-1)  # (B, T, 1)

        # prenet & encoder
        token_feats = self.encPreNet(tokens)
        pos = self.sinusoidal_positional_embed_encoder(token_feats)
        token_feats = torch.add(token_feats, pos) * srcmask_inverted
        attn_mask = (
            srcmask.unsqueeze(-1)
            .repeat(self.enc_num_head, 1, token_feats.shape[1])
            .permute(0, 2, 1)
            .bool()
        )
        token_feats, _ = self.encoder(
            token_feats, src_mask=attn_mask, src_key_padding_mask=srcmask
        )
        token_feats = token_feats * srcmask_inverted    # (B, T, D)
        

        # -------------------- modification area --------------------
        
        B, T, D = token_feats.shape
        speaker_emb = self.speaker_emb(speakers).unsqueeze(1)  # (B, 1, D)
        speaker_emb = speaker_emb.expand(-1, T, -1)  # (B, T, D)

        x = torch.cat(
            [token_feats, speaker_emb, intensity], dim=-1
        )   # (B, T, D + D + 5)
        token_feats = self.concat_proj(x)
        token_feats = token_feats * srcmask_inverted  # (B, T, D)

        # -------------------- modification area end --------------------


        # duration predictor
        predict_durations = self.durPred(token_feats, srcmask_inverted).squeeze(
            -1
        )

        if predict_durations.dim() == 1:
            predict_durations = predict_durations.unsqueeze(0)
        if durations is None:
            dur_pred_reverse_log = torch.clamp(
                torch.special.expm1(predict_durations), 0
            )

        # pitch predictor
        avg_pitch = None
        predict_pitch = self.pitchPred(token_feats, srcmask_inverted)
        # use a pitch rate to adjust the pitch
        predict_pitch = predict_pitch * pitch_rate
        if pitch is not None:
            avg_pitch = average_over_durations(pitch.unsqueeze(1), durations)
            pitch = self.pitchEmbed(avg_pitch)
            avg_pitch = avg_pitch.permute(0, 2, 1)
        else:
            pitch = self.pitchEmbed(predict_pitch.permute(0, 2, 1))
        pitch = pitch.permute(0, 2, 1)
        token_feats = token_feats.add(pitch)

        # energy predictor
        avg_energy = None
        predict_energy = self.energyPred(token_feats, srcmask_inverted)
        # use an energy rate to adjust the energy
        predict_energy = predict_energy * energy_rate
        if energy is not None:
            avg_energy = average_over_durations(energy.unsqueeze(1), durations)
            energy = self.energyEmbed(avg_energy)
            avg_energy = avg_energy.permute(0, 2, 1)
        else:
            energy = self.energyEmbed(predict_energy.permute(0, 2, 1))
        energy = energy.permute(0, 2, 1)
        token_feats = token_feats.add(energy)

        # upsamples the durations
        spec_feats, mel_lens = upsample(
            token_feats,
            durations if durations is not None else dur_pred_reverse_log,
            pace=pace,
        )
        srcmask = get_mask_from_lengths(torch.tensor(mel_lens))
        srcmask = srcmask.to(spec_feats.device)
        srcmask_inverted = (~srcmask).unsqueeze(-1)
        attn_mask = (
            srcmask.unsqueeze(-1)
            .repeat(self.dec_num_head, 1, spec_feats.shape[1])
            .permute(0, 2, 1)
            .bool()
        )

        # decoder
        pos = self.sinusoidal_positional_embed_decoder(spec_feats)
        spec_feats = torch.add(spec_feats, pos) * srcmask_inverted

        output_mel_feats, memory, *_ = self.decoder(
            spec_feats, src_mask=attn_mask, src_key_padding_mask=srcmask
        )

        # postnet
        mel_post = self.linear(output_mel_feats) * srcmask_inverted
        postnet_output = self.postnet(mel_post) + mel_post
        return (
            mel_post,
            postnet_output,
            predict_durations,
            predict_pitch,
            avg_pitch,
            predict_energy,
            avg_energy,
            torch.tensor(mel_lens),
        )