import torch
import torch.nn as nn


########################################################################
# Transformer Encoder Layer with Convolutional Feed-Forward Network
########################################################################
class ConvTransformerEncoderLayer(nn.Module):

    def __init__(
            self, 
            n_heads, 
            hidden_dim, 
            kernel_size, 
            dropout=0.1,
        ):
        super(ConvTransformerEncoderLayer, self).__init__()

        # multi-head attention
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, dropout=dropout, batch_first=True)

        # convolutional feed-forward network
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim * 4, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(hidden_dim * 4, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # activation
        self.activation = nn.GELU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):

        # self-attention
        attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout(attn_output)
        src = self.norm1(src)

        # convolutional feed-forward network
        x = src.transpose(1, 2)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)

        src = src + x
        src = self.norm2(src)
        return src


########################################################################
# Intensity Extractor
########################################################################
class IntensityExtractor(nn.Module):

    def __init__(
            self, 
            n_mels, 
            n_heads, 
            n_emotions,
            n_encoder_layers, 
            hidden_dim, 
            kernel_size, 
            dropout, 
        ):
        super(IntensityExtractor, self).__init__()

        # input projection
        self.input_proj = nn.Linear(n_mels + 2, hidden_dim)

        # FFT blocks
        encoder_layer = ConvTransformerEncoderLayer(n_heads, hidden_dim, kernel_size, dropout)
        self.fft_block = nn.TransformerEncoder(encoder_layer, n_encoder_layers, enable_nested_tensor=False)

        # emotion embedding
        self.emotion_embedding = nn.Embedding(n_emotions, hidden_dim)

        # output linear
        self.classifier = nn.Linear(hidden_dim, n_emotions)


    def forward(self, x, emotions, mask=None):

        H = self.input_proj(x)  # (B, T, H)
        H = self.fft_block(H, src_key_padding_mask=mask)  # (B, T, H)

        emotion_emb = self.emotion_embedding(emotions).unsqueeze(1)
        I = H + emotion_emb  # (B, T, H)

        I = self.classifier(I)  # (B, T, n_emotions)
        
        return I


########################################################################
# Fine-Grained Model
########################################################################
class RankModel(torch.nn.Module):

    def __init__(
            self, 
            n_mels, 
            n_heads,
            n_emotions,
            n_encoder_layers,
            hidden_dim,
            kernel_size,
            dropout,
            **kwargs,
        ):
        super(RankModel, self).__init__()
        self.n_emotions = n_emotions
        
        # intensity extractor
        self.intensity_extractor = IntensityExtractor(n_mels, n_heads, n_emotions, n_encoder_layers, hidden_dim, kernel_size, dropout)

        # ranking score projector
        self.projector = nn.Linear(n_emotions, 1, bias=False)



    def forward(self, emo_X, neu_X, emotions, length, lambdas=None):

        B, C, T = emo_X.size()
        device = emo_X.device

        # create a mask for the transformer
        mask = torch.arange(T, device=device).unsqueeze(0).expand(B, T) \
            >= length.unsqueeze(1)  # (B, T)

        # permute to (B, C, T) -> (B, T, C)
        emo_X = emo_X.permute(0, 2, 1)
        neu_X = neu_X.permute(0, 2, 1)

        # sample mixup weights
        if lambdas is None:
            dist = torch.distributions.Beta(1.0, 1.0)
            lam_i = dist.sample((B,)).to(device).unsqueeze(1).unsqueeze(1)  # (B, 1, 1)
            lam_j = dist.sample((B,)).to(device).unsqueeze(1).unsqueeze(1)  # (B, 1, 1)
        else:
            lam_i = lambdas[0].to(device).unsqueeze(1).unsqueeze(1)
            lam_j = lambdas[1].to(device).unsqueeze(1).unsqueeze(1)

        # mixup at frame level
        Xi_mix = lam_i * emo_X + (1 - lam_i) * neu_X  # (B, T, C)
        Xj_mix = lam_j * emo_X + (1 - lam_j) * neu_X

        # extract intensity features
        Ii = self.intensity_extractor(Xi_mix, emotions, mask=mask)  # (B, T, n_emotions)
        Ij = self.intensity_extractor(Xj_mix, emotions, mask=mask)

        # masked time-average pooling
        Ii = Ii.masked_fill(mask.unsqueeze(-1), 0.0)  # (B, T, n_emotions)
        Ij = Ij.masked_fill(mask.unsqueeze(-1), 0.0) 

        neutral_mask = emotions == 0
        Ii = Ii.masked_fill(neutral_mask.unsqueeze(1).unsqueeze(2), 0.0)
        Ij = Ij.masked_fill(neutral_mask.unsqueeze(1).unsqueeze(2), 0.0)

        hi = Ii.sum(dim=1) / length.unsqueeze(1).float()  # (B, n_emotions)
        hj = Ij.sum(dim=1) / length.unsqueeze(1).float()

        ri = self.projector(hi).squeeze(-1)  # (B,)
        rj = self.projector(hj).squeeze(-1)  # (B,)

        return lam_i, lam_j, Ii, Ij, hi, hj, ri, rj



