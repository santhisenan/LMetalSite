import torch
import torch.nn as nn
import torch.nn.functional as F
from self_attention import *


class MetalSingle(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64, num_encoder_layers=2, num_heads=4, augment_eps=0.05, dropout=0.2):
        super(MetalSingle, self).__init__()

        # Hyperparameters
        self.augment_eps = augment_eps

        # Embedding layers
        self.input_block = nn.Sequential(
                                         nn.LayerNorm(feature_dim, eps=1e-6)
                                        ,nn.Linear(feature_dim, hidden_dim)
                                        ,nn.LeakyReLU()
                                        )

        self.hidden_block = nn.Sequential(
                                          nn.LayerNorm(hidden_dim, eps=1e-6)
                                         ,nn.Dropout(dropout)
                                         ,nn.Linear(hidden_dim, hidden_dim)
                                         ,nn.LeakyReLU()
                                         ,nn.LayerNorm(hidden_dim, eps=1e-6)
                                         )

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_encoder_layers)
        ])

        # output layers
        self.FC_out = nn.Linear(hidden_dim, 1, bias=True)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, protein_feat, mask):
        # Data augmentation
        if self.training and self.augment_eps > 0:
            protein_feat = protein_feat + self.augment_eps * torch.randn_like(protein_feat)

        h_V = self.input_block(protein_feat)
        h_V = self.hidden_block(h_V)

        for layer in self.encoder_layers:
            h_V = layer(h_V, mask)

        logits = self.FC_out(h_V).squeeze(-1) # [B, L]
        return logits


class MetalSite(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64, num_encoder_layers=2, num_heads=4, augment_eps=0.05, dropout=0.2):
        super(MetalSite, self).__init__()

        # Hyperparameters
        self.augment_eps = augment_eps

        # Embedding layers
        self.input_block = nn.Sequential(
                                         nn.LayerNorm(feature_dim, eps=1e-6)
                                        ,nn.Linear(feature_dim, hidden_dim)
                                        ,nn.LeakyReLU()
                                        )

        self.hidden_block = nn.Sequential(
                                          nn.LayerNorm(hidden_dim, eps=1e-6)
                                         ,nn.Dropout(dropout)
                                         ,nn.Linear(hidden_dim, hidden_dim)
                                         ,nn.LeakyReLU()
                                         ,nn.LayerNorm(hidden_dim, eps=1e-6)
                                         )

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_encoder_layers)
        ])

        # ligand-specific layers
        self.FC_ZN1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.FC_ZN2 = nn.Linear(hidden_dim, 1, bias=True)
        self.FC_CA1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.FC_CA2 = nn.Linear(hidden_dim, 1, bias=True)
        self.FC_MG1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.FC_MG2 = nn.Linear(hidden_dim, 1, bias=True)
        self.FC_MN1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.FC_MN2 = nn.Linear(hidden_dim, 1, bias=True)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, protein_feat, mask):
        # Data augmentation
        if self.training and self.augment_eps > 0:
            protein_feat = protein_feat + self.augment_eps * torch.randn_like(protein_feat)

        h_V = self.input_block(protein_feat)
        h_V = self.hidden_block(h_V)

        for layer in self.encoder_layers:
            h_V = layer(h_V, mask)

        logits_ZN = self.FC_ZN2(F.leaky_relu(self.FC_ZN1(h_V))).squeeze(-1) # [B, L]
        logits_CA = self.FC_CA2(F.leaky_relu(self.FC_CA1(h_V))).squeeze(-1) # [B, L]
        logits_MG = self.FC_MG2(F.leaky_relu(self.FC_MG1(h_V))).squeeze(-1) # [B, L]
        logits_MN = self.FC_MN2(F.leaky_relu(self.FC_MN1(h_V))).squeeze(-1) # [B, L]

        logits = torch.cat((logits_ZN, logits_CA, logits_MG, logits_MN), 1)
        return logits
