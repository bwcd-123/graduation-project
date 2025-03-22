import argparse
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
import os
import shutil

import utils
from model import GAT
from evaluation import eva


def delete_files_except(folder_path, files_to_keep):
    # 遍历文件夹中的所有文件和文件夹
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # 如果是文件且不在保留列表中，则删除
        if os.path.isfile(file_path) and filename not in files_to_keep:
            os.remove(file_path)
            print(f"Deleted: {file_path}")


def move_and_rename_file(source_path, destination_folder, new_name):
    """
    将文件移动到目标文件夹并重命名。

    :param source_path: 源文件的完整路径
    :param destination_folder: 目标文件夹的路径
    :param new_name: 文件的新名称（包括扩展名）
    """
    # 确保目标文件夹存在，如果不存在则创建
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 构建目标文件的完整路径
    destination_path = os.path.join(destination_folder, new_name)

    # 移动并重命名文件
    shutil.move(source_path, destination_path)
    print(f"Moved and renamed: {source_path} -> {destination_path}")


def pretrain(dataset):
    model = GAT(
        num_features=args.input_dim,
        hidden_size=args.hidden_size,
        embedding_size=args.embedding_size,
        alpha=args.alpha,
    ).to(device)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # data process
    dataset = utils.data_preprocessing(dataset)
    adj = dataset.adj.to(device)
    adj_label = dataset.adj_label.to(device)
    M = utils.get_M(adj).to(device)

    # data and label
    x = torch.Tensor(dataset.x).to(device)
    y = dataset.y.cpu().numpy()
    best_acc, best_nmi, best_ari, best_f1, best_epoch = 0.0, 0.0, 0.0, 0.0, 0

    for epoch in range(args.max_epoch):
        model.train()
        A_pred, z = model(x, adj, M)
        loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            _, z = model(x, adj, M)
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(
                z.data.cpu().numpy()
            )
            if (epoch+1) % 10 == 0:
                acc, nmi, ari, f1 = eva(y, kmeans.labels_, epoch)
                if acc > best_acc:
                    best_acc = acc
                    best_nmi = nmi
                    best_ari = ari
                    best_f1 = f1
                    best_epoch = epoch+1
                if not os.path.exists("./DAEGC/pretrain"):
                    os.makedirs("./DAEGC/pretrain")
                torch.save(
                    model.state_dict(), f"./DAEGC/pretrain/predaegc_{args.name}_{epoch+1}.pkl"
                )
    print("best_acc: {:.4f}, best_nmi: {:.4f}, best_ari: {:.4f}, best_f1: {:.4f}".format(best_acc, best_nmi, best_ari, best_f1))
    delete_files_except("./DAEGC/pretrain/", [f"predaegc_{args.name}_{best_epoch}.pkl"])
    move_and_rename_file(f"./DAEGC/pretrain/predaegc_{args.name}_{best_epoch}.pkl", "./DAEGC/save", f"a{best_acc:.4f}n{best_nmi:.4f}r{best_ari:.4f}.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--name", type=str, default="Cora")
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_clusters", default=6, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--embedding_size", default=16, type=int)
    parser.add_argument("--weight_decay", type=int, default=5e-3)
    parser.add_argument(
        "--alpha", type=float, default=0.2, help="Alpha for the leaky_relu."
    )
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    datasets = utils.get_dataset(args.name)
    dataset = datasets[0]

    if args.name == "Citeseer":
        args.lr = 0.005
        args.k = None
        args.n_clusters = 6
    elif args.name == "Cora":
        args.lr = 0.001
        args.k = None
        args.n_clusters = 7
    elif args.name == "Pubmed":
        args.lr = 0.001
        args.k = None
        args.n_clusters = 3
    else:
        args.k = None

    args.input_dim = dataset.num_features

    print(args)
    pretrain(dataset)
