from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score, accuracy_score
import torch
import swanlab


def train_one_epoch(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    encode, decode = model(data.x, data.edge_index)
    loss = model.get_sce_loss(decode, data.x)
    loss.backward()
    optimizer.step()
    return loss.item()


def train(model, optimizer, data, args):
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, optimizer, data)
        if args.use_log:
            swanlab.log({'loss':loss})
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss:.4f}")
            model.eval()
            with torch.no_grad():
                encode, _ = model(data.x, data.edge_index)
                cluster_labels = kmeans(encode, k=7)
                acc, ari_score, nmi_score, homogeneity, completeness, v_measure = eval(data.y, cluster_labels)
                if args.use_log:
                    swanlab.log({'acc':acc, 'ari_score':ari_score, 'nmi_score':nmi_score, 'homogeneity':homogeneity, 'completeness':completeness, 'v_measure':v_measure}, step=epoch)


def kmeans(embeddings, k):
    """
    使用 KMeans 聚类

    Parameters:
    embeddings: torch.Tensor, 节点的嵌入
    
    k: int, 聚类的数量
    """
    kmeans = KMeans(n_clusters=k)
    embeddings = embeddings.detach().to('cpu')
    cluster_labels = torch.tensor(kmeans.fit_predict(embeddings))
    return cluster_labels


def eval(true_labels, cluster_labels):
    """
    评估聚类结果

    true_labels: 真实标签

    cluster_labels: 聚类标签
    """
    # 计算ACC、ARI、NMI、同质性、完整性和 V-Measure
    true_labels = true_labels.to('cpu')
    cluster_labels = cluster_labels.to('cpu')
    acc = accuracy_score(true_labels, cluster_labels)
    ari_score = adjusted_rand_score(true_labels, cluster_labels)
    nmi_score = normalized_mutual_info_score(true_labels, cluster_labels)
    homogeneity = homogeneity_score(true_labels, cluster_labels)
    completeness = completeness_score(true_labels, cluster_labels)
    v_measure = v_measure_score(true_labels, cluster_labels)

    print(f"ACC: {acc:.4f}")
    print(f"Adjusted Rand Index: {ari_score:.4f}")
    print(f"Normalized Mutual Information: {nmi_score:.4f}")
    print(f"Homogeneity: {homogeneity:.4f}")
    print(f"Completeness: {completeness:.4f}")
    print(f"V-Measure: {v_measure:.4f}")
    return acc, ari_score, nmi_score, homogeneity, completeness, v_measure
