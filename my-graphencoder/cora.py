import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj
from sklearn.cluster import KMeans
import swanlab
import argparse

from model import StackedEncoder
from utils import pretrain_layers, fine_tune, eval

data_name = 'Cora'
layer_dims = [2708, 1024, 512, 128]
sparsity_params = {'sparsity_target': 0.1, 'sparsity_weight': 0.3}
epochs_per_layer=100
epochs=100

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default=data_name)
    parser.add_argument('--layer_dims', type=list, default=layer_dims)
    parser.add_argument('--sparsity_params', type=dict, default=sparsity_params)
    parser.add_argument('--epochs_per_layer', type=int, default=epochs_per_layer)
    parser.add_argument('--epochs', type=int, default=epochs)
    args = parser.parse_args()
    args.seed=torch.default_generator.seed()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Planetoid(root='../dataset', name=data_name)
    kmeans = KMeans(n_clusters=dataset.num_classes)
    args.kmeans_seed = kmeans.random_state
    data = dataset[0].to(args.device)  # 将数据移到设备

    swanlab.init(
        project=f"graph-encoder-{args.data_name}",
        experiment_name=f"ge-{data_name}",
        config=vars(args),
        mode="cloud"
    )

    # 构建归一化相似矩阵（保持在GPU）
    adj_matrix = to_dense_adj(data.edge_index)[0].squeeze().to(args.device)
    degrees = adj_matrix.sum(dim=1)
    D_inv = torch.diag(1.0 / degrees.clamp(min=1e-12)).to(args.device)
    normalized_S = (D_inv @ adj_matrix).detach()  # 确保不计算梯度

    model = StackedEncoder(layer_dims, sparsity_params).to(args.device)

    # 训练阶段
    model = pretrain_layers(model, normalized_S, epochs_per_layer, args.device)
    final_embedding = fine_tune(model, normalized_S, epochs, args.device)

    # 评估（确保数据在CPU）
    embedding_np = final_embedding.numpy()
    
    labels_pred = kmeans.fit_predict(embedding_np)
    acc, ari_score, nmi_score, homogeneity, completeness, v_measure = eval(data.y.cpu().numpy(), labels_pred)
    swanlab.log({"acc": acc, "ari_score": ari_score, "nmi_score": nmi_score, "homogeneity": homogeneity,
                 "completeness": completeness, "v_measure": v_measure})
    swanlab.finish()
