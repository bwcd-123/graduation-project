import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import GATConv
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

torch.manual_seed(42)

# 1. 数据加载与预处理
dataset = Planetoid(root='../dataset', name='Cora')
data = dataset[0]

# 转换为密集邻接矩阵 (适应论文方法)
adj = to_dense_adj(data.edge_index)[0]  # [num_nodes, num_nodes]
features = data.x  # [num_nodes, 1433]
labels = data.y    # [num_nodes]

# 添加自环并对称归一化
def preprocess_adj(adj):
    adj = adj + torch.eye(adj.size(0))
    row_sum = adj.sum(1).clamp(min=1e-12).pow(-0.5)
    return row_sum.unsqueeze(1) * adj * row_sum.unsqueeze(0)

adj = preprocess_adj(adj)
features = F.normalize(features, p=2, dim=1)  # L2归一化

# 2. 模型定义
class GraphCluster(nn.Module):
    def __init__(self, in_dim=1433, hid_dim=256, heads=8, num_classes=7):
        super().__init__()
        # 编码器
        self.gat1 = GATConv(in_dim, hid_dim, heads=heads)
        self.gat2 = GATConv(hid_dim*heads, hid_dim, heads=1)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(hid_dim, 512),
            nn.ELU(),
            nn.Linear(512, in_dim)
        )
        
        # 聚类中心初始化
        self.cluster_centers = nn.Parameter(torch.Tensor(num_classes, hid_dim))
        nn.init.xavier_normal_(self.cluster_centers)
        
    def forward(self, x, edge_index, mask_rate=0.3):
        # 随机掩码 (每次forward不同)
        mask = torch.randperm(x.size(0))[:int(x.size(0)*mask_rate)]
        x_masked = x.clone()
        if self.training:
            x_masked[mask] = 0
        
        # 双分支编码
        h1 = F.elu(self.gat1(x, edge_index))
        h1 = self.gat2(h1, edge_index)
        
        h2 = F.elu(self.gat1(x_masked, edge_index))
        h2 = self.gat2(h2, edge_index)
        
        # 动态权重融合
        alpha = torch.sigmoid(torch.mean(h1 * h2, dim=1, keepdim=True))
        fused = alpha * h1 + (1 - alpha) * h2
        
        # 重建特征
        x_recon = self.decoder(fused)
        
        # 计算聚类分布
        dist = (fused.unsqueeze(1) - self.cluster_centers).pow(2).sum(-1)  # [N, K]
        q = 1.0 / (1.0 + dist)
        q = F.softmax(q, dim=1)
        
        return fused, x_recon, q

# 3. 损失函数
class ClusterLoss(nn.Module):
    def __init__(self, epsilon=10.0, gamma=1.0, device='cpu'):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = gamma
        self.device = device
        
    def forward(self, x_recon, x_real, q, adj_recon, adj_real):
        # 特征重建损失 (余弦相似度)
        cosine_loss = (1 - F.cosine_similarity(x_recon, x_real, dim=1)).pow(self.gamma).mean()
        
        # 结构重建损失 (带权重的BCE)
        pos_weight = torch.tensor([adj_real.size(0)**2 / (adj_real.sum() * 2)]).to(self.device)
        bce_loss = F.binary_cross_entropy_with_logits(adj_recon, adj_real, pos_weight=pos_weight)
        
        # 聚类损失
        p = (q**2) / q.sum(0)
        p = (p.t() / p.sum(1)).t().detach()
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        
        return cosine_loss + bce_loss + self.epsilon * kl_loss

# 4. 训练流程
def train(model, features, adj, edge_index, labels, epochs=500, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 移动所有数据到设备
    model = model.to(device)
    features = features.to(device)
    adj = adj.to(device)
    edge_index = edge_index.to(device)
    labels = labels.to(device)  # 如果使用GPU计算acc时需要
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = ClusterLoss(epsilon=10.0, gamma=1.0, device=device)
    
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        _, x_recon, q = model(features, edge_index)
        adj_recon = torch.sigmoid(torch.mm(x_recon, x_recon.t())).to(device)
        
        # 确保adj在正确设备
        loss = criterion(x_recon, features, q, adj_recon, adj)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 每10轮评估
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                _, _, q = model(features, edge_index)
                y_pred = q.argmax(1).cpu().numpy()  # 移回CPU用于sklearn计算
                acc = aligned_accuracy(labels.cpu().numpy(), y_pred)
                
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), 'best_model.pth')
                    
            print(f'Epoch {epoch:03d} | Loss: {loss.item():.4f} | Acc: {acc:.4f}')
    
    print(f'Best Accuracy: {best_acc:.4f}')
    return best_acc


# 对齐准确率计算
def aligned_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(cm, maximize=True)
    return cm[row_ind, col_ind].sum() / len(y_true)

# 运行训练
model = GraphCluster(in_dim=1433, hid_dim=256, heads=8, num_classes=7)
train(model, features, adj, data.edge_index, labels, epochs=600, lr=1e-3)