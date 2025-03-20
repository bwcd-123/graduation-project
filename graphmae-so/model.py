import torch
import torch.nn as nn
from torch.nn import BatchNorm1d
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
    """
    get normalization layer

    Parameters
    ----------
    norm : str
        normalization function
    num_features : int
        number of features
    """
    if norm == 'batchnorm':
        return BatchNorm1d(num_features)
    

def get_encoder(args):
    """
    get encoder model
    """
    if args.encoder == 'gcn':
        return GCN(args.layer_de, args.features, args.encode_size, args.middle, args.activation, args.norm)
    elif args.encoder == 'gat':
        return GAT(args.layer_de, args.features, args.encode_size, args.middle, args.activation, args.norm)
    else:
        raise NotImplementedError
    

def get_decoder(args):
    """
    get decoder model
    """
    if args.decoder == 'gcn':
        return GCN(args.layer_de, args.encode_size, args.features, args.middle, args.activation, args.norm)
    elif args.decoder == 'gat':
        return GAT(args.layer_de, args.encode_size, args.features, args.middle, args.activation, args.norm)
    else:
        raise NotImplementedError


def get_optimizer(args, model):
    """
    get optimizer
    """
    if args.optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError


class GCN(nn.Module):
    def __init__(self, num_layers, input, output, middle=None, activation='leakyrelu', norm=None, *args, **kwargs):
        """
        create a GCN model

        Parameters
        ----------
        num_layers : int
            number of layers
        input : int
            input dim
        middle : int
            middle dim
        output : int
            output dim
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
        if num_layers == 1:
            self.convs = nn.ModuleList()
            self.convs.append(
                GCNConv(input, output))
        else:
            self.convs = nn.ModuleList()
            fea_list = [input] + [middle] * (num_layers-1) + [output]
            for i in range(num_layers):
                act_norm = nn.Sequential()
                if norm is not None and i != 0:
                    act_norm.append(
                        get_norm(norm, num_features=middle))
                act_norm.append(
                    get_activation(activation))
                self.convs.append(act_norm)
                self.convs.append(
                    GCNConv(fea_list[i], fea_list[i+1]))
        
    def forward(self, x, edge_index):
        for i in range(len(self.convs)):
            if i % 2 == 0:
                x = self.convs[i](x, edge_index)
            else:
                x = self.convs[i](x)
        return x

class GAT(nn.Module):
    def __init__(self, num_layers, input, output, middle=None, activation='leakyrelu', norm=None, *args, **kwargs):
        """
        create a GAT model

        Parameters
        ----------
        num_layers : int
            number of layers
        input : int
            input dim
        middle : int
            middle dim
        output : int
            output dim
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
        if num_layers == 1:
            self.convs = nn.ModuleList()
            self.convs.append(
                GATConv(input, output))
        else:
            self.convs = nn.ModuleList()
            fea_list = [input] + [middle] * (num_layers-1) + [output]
            for i in range(num_layers):
                if i != 0:
                    act_norm = nn.Sequential()
                    if norm is not None:
                        act_norm.append(
                            get_norm(norm, num_features=middle))
                    act_norm.append(
                        get_activation(activation))
                    self.convs.append(act_norm)
                self.convs.append(
                    GATConv(fea_list[i], fea_list[i+1]))

    def forward(self, x, edge_index):
        for i in range(len(self.convs)):
            if i % 2 == 0:
                x = self.convs[i](x, edge_index)
            else:
                x = self.convs[i](x)
        return x


class GraphMAE(GAE):
    def __init__(self, encoder, decoder, encode_size, num_classes, mask_ratio=0.5):
        super(GraphMAE, self).__init__(encoder, decoder)
        self.mask_ratio = mask_ratio
        self.centers = nn.Parameter(torch.zeros([num_classes, encode_size]))
        nn.init.xavier_uniform_(self.centers.data, gain=1.414)

    def forward(self, x, edge_index):
        # mask the input features
        mask_num = int(x.size(0) * self.mask_ratio)
        mask_index = torch.randperm(x.size(0))[:mask_num].to(x.device)
        mask_x = x.clone()
        if self.training:
            mask_x[mask_index] = 0

        encoded = self.encode(mask_x, edge_index)
        q = 1.0 / (1.0 + (encoded.unsqueeze(1) - self.centers).pow(2).sum(dim=2))
        q = q.pow((1 + 1.0) / 2.0)
        q = q / q.sum(dim=1, keepdim=True)
        mask_encoded = encoded.clone()
        if self.training:
            mask_encoded[mask_index] = 0
        decoded = self.decode(mask_encoded, edge_index)
        return q, decoded

    def get_sce_loss(self, decoded, true_x, gamma=1):
        return ((1-F.cosine_similarity(decoded, true_x))**gamma).sum() / int(self.mask_ratio*len(true_x))
    
    def get_kl_loss(self, q):
        p = q**2 / q.sum(dim=0)
        p = p / p.sum(dim=1, keepdim=True)
        loss_clu = F.kl_div(q, p.detach(), reduction='batchmean')
        return loss_clu
    
    def __str__(self):
        return super().__str__() + f' with mask ratio {self.mask_ratio}'
    