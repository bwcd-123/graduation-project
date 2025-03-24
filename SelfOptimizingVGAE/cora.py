import torch
from torch_geometric.datasets import Planetoid
import swanlab
import argparse

from model import SelfOptimizingVGAE
from utils import predict, eval, train

seed = torch.default_generator.seed()
# seed = 40473483797100 # no kaiming initial cora
# seed = 42739336062500 # with kaiming initial cora
# torch.manual_seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Citeseer')
    parser.add_argument('--channels', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.seed = torch.default_generator.seed()

    swanlab.init(
        project=f'SelfOptimizingVGAE-{args.dataset}',
        description=f'SelfOptimizingVGAE on {args.dataset}',
        config=vars(args),
        mode='cloud'
    )
    dataset = Planetoid(root='../dataset', name=args.dataset)
    data = dataset[0]
    model = SelfOptimizingVGAE(dataset.num_features, args.channels, dataset.num_classes).to(args.device)
    data = data.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train(model, data, optimizer, args.epochs)
    cluster_labels = predict(model, data)
    ret = eval(data.y.cpu().numpy(), cluster_labels.cpu().numpy())
    swanlab.log(ret)
    swanlab.finish()
