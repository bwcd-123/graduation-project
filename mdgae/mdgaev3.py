import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score, accuracy_score
from sklearn.cluster import KMeans


class GraphCluster(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_clusters: int):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim)
        self.gat2 = GATConv(hidden_dim, n_clusters)
        self.cluster_centers = None

    def init_cluster_centers(self, features: torch.Tensor, edge_index: torch.Tensor):
        """使用实际数据初始化聚类中心"""
        with torch.no_grad():  # 禁用梯度计算
            h = F.elu(self.gat1(features, edge_index))
            h = F.elu(self.gat2(h, edge_index))
            self.cluster_centers = h.detach()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """模型前向传播"""
        h = F.elu(self.gat1(x, edge_index))
        return self.gat2(h, edge_index)

def load_data():
    """含模拟标签的数据加载"""
    dataset = Planetoid(root='../dataset', name='Cora')
    data = dataset[0]
    return data

def evaluate(outputs: torch.Tensor, labels: torch.Tensor) -> dict:
    """聚类效果评估"""
    embeddings = outputs.detach().cpu().numpy()
    true_labels = labels.cpu().numpy()
    
    # KMeans聚类
    kmeans = KMeans(n_clusters=10, random_state=0).fit(embeddings)
    pred_labels = kmeans.labels_
    
    grade = {
        'ACC': accuracy_score(true_labels, pred_labels),
        'NMI': normalized_mutual_info_score(true_labels, pred_labels),
        'ARI': adjusted_rand_score(true_labels, pred_labels)
    }
    print(grade)
    # 计算指标
    return grade

def train():
    # 数据加载
    data = load_data()
    features, edge_index, labels = data['x'], data['edge_index'], data['y']
    
    # 模型初始化
    model = GraphCluster(in_dim=1433, hidden_dim=32, n_clusters=7)
    model.init_cluster_centers(features, edge_index)
    
    # 训练准备
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_nmi = 0.0
    
    for epoch in range(100):
        # 训练步骤
        model.train()
        optimizer.zero_grad()
        outputs = model(features, edge_index)
        loss = F.mse_loss(outputs, model.cluster_centers)
        loss.backward()
        optimizer.step()
        
        # 每10个epoch评估一次
        if (epoch+1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                eval_outputs = model(features, edge_index)
                metrics = evaluate(eval_outputs, labels)
                
                # 保存最佳模型
                if metrics['NMI'] > best_nmi:
                    best_nmi = metrics['NMI']
                    torch.save(model.state_dict(), 'best_model.pth')
                
                # print(f"Epoch {epoch+1:03d} | "
                #       f"Loss: {loss.item():.4f} | "
                #       f"NMI: {metrics['NMI']:.4f} | "
                #       f"ARI: {metrics['ARI']:.4f}")

if __name__ == "__main__":
    train()