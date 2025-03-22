import torch
from torch_geometric.datasets import Planetoid
import argparse
import swanlab

from model import VGAEModel, kmeans, eval


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="Cora")
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--channels', type=int, default=16)
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.seed = torch.default_generator.seed()
    swanlab.init(
        project=f'vgae-{args.dataset}',
        config=vars(args),
        mode='cloud'
    )

    dataset = Planetoid(root='../dataset', name=args.dataset)
    data = dataset[0]
    model = VGAEModel(dataset.num_features, args.channels).to(args.device)
    data = data.to(args.device)

    model.pretrain(data, args.epochs, args.lr)
    embeddings = model.evaluate(data).cpu().numpy()
    cluster_labels = kmeans(embeddings, dataset.num_classes)
    acc, ari, nmi, homogeneity, completeness, v_measure = eval(cluster_labels, data.y.cpu().numpy())
    swanlab.log({
        "acc": acc,
        "nmi": nmi,
        "ari": ari,
        "homogeneity": homogeneity,
        'v_measure':v_measure
        })