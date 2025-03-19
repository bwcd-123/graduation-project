import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj
from sklearn.cluster import KMeans
import swanlab

from model import StackedEncoder
from utils import pretrain_layers, fine_tune, eval

data_root = '../dataset'
data_name = 'cora'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid(root=data_root, name=data_name)
data = dataset[0].to(device)  # 将数据移到设备
layer_dims = [2708, 1024, 512, 128]
sparsity_params = {'sparsity_target': 0.1, 'sparsity_weight': 0.3}
epochs_per_layer=100
epochs=100
random_state=42

torch.manual_seed(random_state)
swanlab.init(
    project="graph-encoder",
    experiment_name=f"ge-{data_name}",
    config={
        'layer_dims': layer_dims,
        'sparsity_params': sparsity_params,
        'data_root': data_root,
        'data_name': data_name,
        'epochs_per_layer': epochs_per_layer,
        'epochs': epochs,
        'random_state': random_state,
        'seed': random_state,
        'device': device
    },
    mode="cloud"
)

# 构建归一化相似矩阵（保持在GPU）
adj_matrix = to_dense_adj(data.edge_index)[0].squeeze().to(device)
degrees = adj_matrix.sum(dim=1)
D_inv = torch.diag(1.0 / degrees.clamp(min=1e-12)).to(device)
normalized_S = (D_inv @ adj_matrix).detach()  # 确保不计算梯度

model = StackedEncoder(layer_dims, sparsity_params).to(device)

# 训练阶段
model = pretrain_layers(model, normalized_S, epochs_per_layer, device)
final_embedding = fine_tune(model, normalized_S, epochs, device)

# 评估（确保数据在CPU）
embedding_np = final_embedding.numpy()
kmeans = KMeans(n_clusters=dataset.num_classes, random_state=random_state)
labels_pred = kmeans.fit_predict(embedding_np)
eval(data.y.cpu().numpy(), labels_pred)
