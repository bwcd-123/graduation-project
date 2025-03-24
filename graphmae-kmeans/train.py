import argparse
import torch
from torch_geometric.datasets import Planetoid
import swanlab

from model import GraphMAE, get_encoder, get_decoder, get_optimizer
from utils import train


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphMAE Training')
    parser.add_argument('--dataset', type=str, default='Pubmed', help='dataset name')
    parser.add_argument('--root', type=str, default='../dataset', help='data root')
    parser.add_argument('--encoder', type=str, default='gat', help='encoder name')
    parser.add_argument('--layer_en', type=int, default=2, help='number of encoder layers')
    parser.add_argument('--decoder', type=str, default='gat', help='decoder name')
    parser.add_argument('--layer_de', type=int, default=2, help='number of decoder layers')
    parser.add_argument('--middle', type=int, default=256, help='middle layer size')
    parser.add_argument('--encode_size', type=int, default=16, help='encode size')
    parser.add_argument('--activation', type=str, default='leakyrelu', help='activation function')
    parser.add_argument('--norm', type=str, default=None, help='normalization layer')
    parser.add_argument('--mask_ratio', type=float, default=0.5, help='mask ratio')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('--no_log', action='store_true', help='use log to record performance')
    parser.add_argument('--excute_times', type=int, default=100, help='number of excute times')
    args = parser.parse_args()
    for i in range(args.excute_times):
        print(f'Excute time {i + 1}')
        dataset = Planetoid(root=args.root, name=args.dataset)
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        args.seed = torch.default_generator.seed()
        args.features = dataset.num_features
        
        if not args.no_log:
            swanlab.init(
                project=f'graphmae-kmeans-{args.dataset}',
                config=vars(args),
                mode='cloud'
            )
        
        data = dataset[0].to(args.device)

        encoder = get_encoder(args).to(args.device)
        decoder = get_decoder(args).to(args.device)
        model = GraphMAE(encoder, decoder, mask_ratio=args.mask_ratio).to(args.device)
        print(model)
        optimizer = get_optimizer(args, model)

        train(model, optimizer, data, args)

        if not args.no_log:
            swanlab.finish()