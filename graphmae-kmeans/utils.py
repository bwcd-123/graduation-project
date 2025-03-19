from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score, accuracy_score


def train_one_epoch(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    encode, decode = model(data.x, data.edge_index)
    loss = model.get_sce_loss(decode, data.x)
    loss.backward()
    optimizer.step()
    return loss.item()


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
