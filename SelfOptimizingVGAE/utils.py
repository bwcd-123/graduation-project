from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score, accuracy_score
import torch.nn.functional as F
import torch
import swanlab

# old model
def train_one_epoch(model, data, optmizer, kl_thre=0.5):
    """
    训练模型

    model: 模型

    data: 数据

    optmizer: 优化器

    epochs: 训练轮数

    device: 训练设备

    verbose: 是否打印训练信息

    kl_thre: KL散度阈值
    """
    model.train()
    logits, q = model(data.x, data.edge_index)
    # print(f"logits: {logits.shape}, q: {q.shape}")
    # 联合损失：分类交叉熵 + KL散度聚类损失[2](@ref)
    loss_cls = F.nll_loss(logits[data.train_mask], data.y[data.train_mask])
    # 目标分布计算（强化高置信度分配）
    p = q**2 / q.sum(dim=0)
    p = p / p.sum(dim=1, keepdim=True)
    loss_clu = F.kl_div(q.log(), p.detach(), reduction='batchmean')
    total_loss = loss_cls + kl_thre * loss_clu
    optmizer.zero_grad()
    total_loss.backward()
    optmizer.step()
    return total_loss.item()


def train(model, data, opitmizer, epochs, verbose=True, kl_thre=0.5):
    """
    训练模型

    model: 模型

    data: 数据

    optmizer: 优化器

    epochs: 训练轮数

    verbose: 是否打印训练信息

    kl_thre: KL散度阈值
    """
    for epoch in range(epochs):
        loss = train_one_epoch(model, data, opitmizer, kl_thre)
        swanlab.log({"loss": loss})
        if (epoch+1) % 10 == 0 and verbose: print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
    

def predict(model, data):
    """
    预测模型

    model: 模型

    data: 数据
    """
    model.eval()
    with torch.no_grad():
        _, q = model(data.x, data.edge_index)
    return q.argmax(dim=1)


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
    return {
        "acc": acc,
        "ari": ari_score,
        "nmi": nmi_score,
        "homogeneity": homogeneity,
        "completeness": completeness,
        "v_measure": v_measure,
    }
