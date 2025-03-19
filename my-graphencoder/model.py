import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, output_dim, sparsity_target=0.1, sparsity_weight=0.5):
        super().__init__()
        self.encoder = nn.Linear(input_dim, output_dim)
        self.decoder = nn.Linear(output_dim, input_dim)
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight

    def forward(self, x):
        h = torch.sigmoid(self.encoder(x))
        x_recon = torch.sigmoid(self.decoder(h))
        return x_recon, h

    def sparse_loss(self, h):
        avg_activation = h.mean(dim=0)
        kl_div = self.sparsity_target * torch.log(self.sparsity_target / avg_activation) + \
                 (1 - self.sparsity_target) * torch.log((1 - self.sparsity_target) / (1 - avg_activation))
        return kl_div.mean()

class StackedEncoder(nn.Module):
    def __init__(self, layer_dims, sparsity_params):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_dims)-1):
            self.layers.append(
                SparseAutoencoder(layer_dims[i], layer_dims[i+1], **sparsity_params)
            )
    
    def forward(self, x, return_all=False):
        hidden = x
        all_outputs = []
        for layer in self.layers:
            x_recon, hidden = layer(hidden)
            all_outputs.append( (x_recon, hidden) )
        
        if return_all:
            return hidden, all_outputs
        else:
            return hidden
