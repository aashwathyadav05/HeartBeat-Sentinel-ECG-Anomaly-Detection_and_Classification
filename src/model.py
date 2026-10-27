import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self, seq_len: int = 140, latent_dim: int = 32):
        super(ConvAutoencoder, self).__init__()
        self.seq_len = seq_len
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        # Compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, seq_len)
            enc_out = self.encoder(dummy)
            self.enc_shape = enc_out.shape  # (1, 64, L')
            self.flat_dim = enc_out.numel()

        self.fc_enc = nn.Linear(self.flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flat_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        # x: (B,1,L)
        x_enc = self.encoder(x)
        x_flat = x_enc.view(x_enc.size(0), -1)
        z = self.fc_enc(x_flat)
        x_flat_dec = self.fc_dec(z)
        x_unflat = x_flat_dec.view(x_enc.size(0), x_enc.size(1), x_enc.size(2))
        x_rec = self.decoder(x_unflat)
        # Adjust length if necessary to match input
        if x_rec.shape[2] != x.shape[2]:
            x_rec = torch.nn.functional.interpolate(x_rec, size=x.shape[2], mode='linear', align_corners=False)
        return x_rec


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
