import argparse
import torch
from torch_geometric.datasets import Planetoid

from model_kmeans import GraphMAE, get_encoder, get_decoder


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphMAE Training')
    parser.add_argument('--dataset', type=str, default='Cora', help='dataset name')
    parser.add_argument('--root', type=str, default='../dataset', help='data root')
    parser.add_argument('--encoder', type=str, default='GAT', help='encoder name')
    parser.add_argument('--layer_en', type=int, default=2, help='number of encoder layers')
    parser.add_argument('--decoder', type=str, default='GAT', help='decoder name')
    parser.add_argument('--layer_de', type=int, default=2, help='number of decoder layers')
    parser.add_argument('--middle_list', type=str, default='256', help='middle layer size. If multiple, use comma to separate')
    parser.add_argument('--activation', type=str, default='leakyrelu', help='activation function')
    parser.add_argument('--norm', type=str, default='batchnorm', help='normalization layer')
    parser.add_argument('--mask_ratio', type=float, default=0.5, help='mask ratio')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    args = parser.parse_args()
    args.middle_list = [int(i) for i in args.middle_list.split(',')]
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = Planetoid(root=args.root, name=args.dataset)
    data = dataset[0].to(args.device)

    encoder = get_encoder(args).to(args.device)
    decoder = get_decoder(args).to(args.device)
    model = GraphMAE(encoder, decoder, mask_ratio=args.mask_ratio).to(args.device)