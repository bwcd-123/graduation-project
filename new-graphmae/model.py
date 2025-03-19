from functools import partial
import torch
import torch.nn as nn
from torch.nn import BatchNorm1d as BatchNorm
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GAE


def get_activation(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'leakyrelu':
        return nn.LeakyReLU()
    elif activation == 'elu':
        return nn.ELU()
    elif activation == 'selu':
        return nn.SELU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise NotImplementedError


def get_norm(norm, num_features):
    if norm == 'batchnorm':
        return BatchNorm(num_features)
    else:
        raise NotImplementedError


class GCN(nn.Module):
    def __init__(self, num_layers, fea_list, activation='leakyrelu', norm=None, *args, **kwargs):
        """
        create a GCN model

        Parameters
        ----------
        num_layers : int
            number of layers
        fea_list : list
            list of feature size for each layer
        activation : str
            activation function
        norm : str
            normalization function
        *args : list
            arguments for activation function
        **kwargs : dict
            arguments for normalization function
        """
        super(GCN, self).__init__(*args, **kwargs)
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i != 0:
                act_norm = nn.Sequential()
                if norm is not None:
                    act_norm.append(get_norm(norm, num_features=fea_list[i]))
                act_norm.append(get_activation(activation))
                self.convs.append(act_norm)
            self.convs.append(GCNConv(fea_list[i], fea_list[i+1]))
        
    def forward(self, x, edge_index):
        for i in range(len(self.convs)):
            if i % 2 == 0:
                x = self.convs[i](x, edge_index)
            else:
                x = self.convs[i](x)
        return x

class GAT(nn.Module):
    def __init__(self, num_layers, fea_list, activation='leakyrelu', norm=None, *args, **kwargs):
        """
        create a GAT model

        Parameters
        ----------
        num_layers : int
            number of layers
        fea_list : list
            list of feature size for each layer
        activation : str
            activation function
        norm : str
            normalization function
        *args : list
            arguments for activation function
        **kwargs : dict
            arguments for normalization function
        """
        super(GAT, self).__init__(*args, **kwargs)
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i != 0:
                act_norm = nn.Sequential()
                if norm is not None:
                    act_norm.append(get_norm(norm, fea_list[i]))
                act_norm.append(get_activation(activation))
                self.convs.append(act_norm)
            self.convs.append(GATConv(fea_list[i], fea_list[i+1]))
        
    def forward(self, x, edge_index):
        for i in range(len(self.convs)):
            if i % 2 == 0:
                x = self.convs[i](x, edge_index)
            else:
                x = self.convs[i](x)
        return x


class GraphMAE(GAE):
    def __init__(self, encoder, decoder, mask_ratio=0.5):
        super(GraphMAE, self).__init__(encoder, decoder)
        self.mask_ratio = mask_ratio

    def forward(self, x, edge_index):
        # mask the input features
        mask_num = int(x.size(0) * self.mask_ratio)
        mask_index = torch.randperm(x.size(0))[:mask_num].to(x.device)
        mask_x = x.clone()
        if self.training:
            mask_x[mask_index] = 0

        encoded = self.encode(mask_x, edge_index)
        mask_encoded = encoded.clone()
        if self.training:
            mask_encoded[mask_index] = 0
        decoded = self.decode(mask_encoded, edge_index)
        return encoded, decoded

    def get_sce_loss(self, decoded, true_x, gamma=1):
        return ((1-F.cosine_similarity(decoded, true_x))**gamma).sum() / int(self.mask_ratio*len(true_x))
    
    def __str__(self):
        return super().__str__() + f' with mask ratio {self.mask_ratio}'
    