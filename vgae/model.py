# @misc{kipf2016variationalgraphautoencoders,
#   title = {Variational Graph Auto-Encoders},
#   author = {Kipf, Thomas N. and Welling, Max},
#   year = {2016},
#   eprint = {1611.07308},
#   archiveprefix = {arXiv},
#   langid = {american},
#   keywords = {GAE,VGAE},
# }


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, VGAE
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score, accuracy_score
import swanlab


# 定义 GCN 编码器
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, p=0.5):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.batch = torch.nn.BatchNorm1d(2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logvar = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(F.dropout(self.conv1(x, edge_index), training=self.training))
        x = F.relu(self.conv2(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)

# 定义 VGAE 模型
class VGAEModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGAEModel, self).__init__()
        self.encoder = GCNEncoder(in_channels, out_channels)
        self.vgae = VGAE(self.encoder)

    def encode(self, x, edge_index):
        return self.vgae.encode(x, edge_index)

    def reparameterize(self, mu, logvar):
        return self.vgae.reparameterize(mu, logvar)

    def pretrain(self, data, epochs, lr):
        """
        训练模型
        """
        optimizer = torch.optim.Adam(self.parameters(), lr)
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            z = self.encode(data.x, data.edge_index)
            loss = self.vgae.recon_loss(z, data.edge_index)
            loss = loss + (1 / data.num_nodes) * self.vgae.kl_loss()
            loss.backward()
            optimizer.step()
            swanlab.log({"loss": loss.item()})
            # if (epoch+1) % 10 == 0:
            #     print(f'Epoch: {epoch+1}, Loss: {loss:.4f}')

    def evaluate(self, data):
        """
        评估模型
        """
        self.eval()
        with torch.no_grad():
            z = self.encode(data.x, data.edge_index)
        return z

def kmeans(embeddings, k):
    """
    使用 KMeans 聚类

    Parameters:
    embeddings: numpy.ndarray, 节点的嵌入
    
    k: int, 聚类的数量
    """
    kmeans = KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels


def eval(true_labels, cluster_labels):
    """
    评估聚类结果

    true_labels: 真实标签

    cluster_labels: 聚类标签
    """
    # 计算ACC、ARI、NMI、同质性、完整性和 V-Measure
    acc = accuracy_score(true_labels, cluster_labels)
    ari_score = adjusted_rand_score(true_labels, cluster_labels)
    nmi_score = normalized_mutual_info_score(true_labels, cluster_labels)
    homogeneity = homogeneity_score(true_labels, cluster_labels)
    completeness = completeness_score(true_labels, cluster_labels)
    v_measure = v_measure_score(true_labels, cluster_labels)

    print(f"ACC: {acc}")
    print(f"Adjusted Rand Index: {ari_score}")
    print(f"Normalized Mutual Information: {nmi_score}")
    print(f"Homogeneity: {homogeneity}")
    print(f"Completeness: {completeness}")
    print(f"V-Measure: {v_measure}")
    return acc, ari_score, nmi_score, homogeneity, completeness, v_measure
