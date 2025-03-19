# @misc{kipf2016variationalgraphautoencoders,
#   title = {Variational Graph Auto-Encoders},
#   author = {Kipf, Thomas N. and Welling, Max},
#   year = {2016},
#   eprint = {1611.07308},
#   archiveprefix = {arXiv},
#   langid = {american},
#   keywords = {GAE,VGAE},
# }


# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, VGAE


# 定义 GCN 编码器
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# 自优化聚类模型定义
class SelfOptimizingVGAE(torch.nn.Module):
    def __init__(self, num_features, middle_dim, num_clusters):
        super().__init__()
        self.conv1 = GCNConv(num_features, middle_dim)
        self.conv2 = GCNConv(middle_dim, num_clusters)
        self.cluster_centers = torch.nn.Parameter(torch.rand(num_clusters, middle_dim))
        torch.nn.init.kaiming_uniform_(self.cluster_centers)

    def forward(self, x, edge_index):
        # 图卷积层
        x = F.relu(self.conv1(x, edge_index))
        embeddings = x  # 保存嵌入用于聚类
        x = self.conv2(x, edge_index)
        
        # 软分配计算（Student's t分布）
        q = 1.0 / (1.0 + (embeddings.unsqueeze(1) - self.cluster_centers).pow(2).sum(dim=2))
        q = q.pow((1 + 1.0) / 2.0)
        q = q / q.sum(dim=1, keepdim=True)  # 归一化为概率分布[2](@ref)
        return F.log_softmax(x, dim=1), q
