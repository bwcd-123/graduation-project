import argparse

from sklearn.cluster import KMeans

import torch
import torch.nn.functional as F
from torch.optim import Adam
import swanlab

import utils
from model import DAEGC
from evaluation import eva


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def trainer(dataset):
    model = DAEGC(num_features=args.input_dim, hidden_size=args.hidden_size,
                  embedding_size=args.embedding_size, alpha=args.alpha, num_clusters=args.n_clusters).to(device)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # data process
    dataset = utils.data_preprocessing(dataset)
    adj = dataset.adj.to(device)
    adj_label = dataset.adj_label.to(device)
    M = utils.get_M(adj).to(device)

    # data and label
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y.cpu().numpy()

    # get kmeans and pretrain cluster resul t
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

    for epoch in range(args.max_epoch):
        model.train()
        A_pred, z, q = model(data, adj, M)
        if epoch % args.update_interval == 0:
            # update_interval            
            Q = q.detach().data.cpu().numpy().argmax(1)  # Q
            acc, nmi, ari, f1 = eva(y, Q, epoch)
            swanlab.log({"acc": acc, "nmi": nmi, "ari": ari, "f1": f1})
        p = target_distribution(q.detach())

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        re_loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))

        loss = 10 * kl_loss + re_loss
        swanlab.log({"kl_loss": kl_loss.item(), "re_loss": re_loss.item(), "loss": loss.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# DAEGC/daegc.py --name Pubmed --max_epoch 10
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='Pubmed')
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--n_clusters', default=6, type=int)
    parser.add_argument('--update_interval', default=1, type=int)  # [1,3,5]
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--embedding_size', default=16, type=int)
    parser.add_argument('--weight_decay', type=int, default=5e-3)
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    args = parser.parse_args()
    args.seed = torch.default_generator.seed()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")
    # device = "cpu"

    datasets = utils.get_dataset(args.name)
    dataset = datasets[0]

    if args.name == 'Citeseer':
      args.lr = 0.001
      args.k = None
      args.n_clusters = 6
    elif args.name == 'Cora':
      args.lr = 0.001
      args.k = None
      args.n_clusters = 7
    elif args.name == "Pubmed":
        args.lr = 0.001
        args.k = None
        args.n_clusters = 3
    else:
        args.k = None
        
    swanlab.init(
        project=f"daegc-{args.name}",
        config=vars(args),
        mode='cloud'
    )
    
    args.pretrain_path = f'./pretrain/predaegc_{args.name}_{args.epoch}.pkl'
    args.input_dim = dataset.num_features


    print(args)
    trainer(dataset)
    swanlab.finish()
