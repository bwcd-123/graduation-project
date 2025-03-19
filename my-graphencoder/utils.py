import torch
import torch.nn.functional as F
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score, accuracy_score
import swanlab


def pretrain_layers(model, X, epochs_per_layer=50, device='cpu'):
    model = model.to(device)
    current_input = X.clone().to(device)
    
    for layer_idx, layer in enumerate(model.layers):
        print(f"Pretraining Layer {layer_idx+1}/{len(model.layers)}")
        
        # 冻结之前所有层
        for prev_layer in model.layers[:layer_idx]:
            for param in prev_layer.parameters():
                param.requires_grad_(False)
        
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.001)
        
        for epoch in range(epochs_per_layer):
            optimizer.zero_grad()
            x_recon, h = layer(current_input)
            
            # 确保计算在相同设备
            recon_loss = F.mse_loss(x_recon, current_input)
            sparse_loss = layer.sparse_loss(h)
            total_loss = recon_loss + layer.sparsity_weight * sparse_loss
            
            total_loss.backward()
            optimizer.step()
            swanlab.log({f"{layer_idx}-loss": total_loss.item()})
            
            if (epoch+1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs_per_layer} Loss: {total_loss.item():.4f}")
        
        # 更新输入并保持设备一致
        with torch.no_grad():
            _, current_input = layer(current_input)
            current_input = current_input.detach().to(device)
    
    return model

def fine_tune(model, X, epochs=100, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 前向传播
        final_emb, all_layer_outputs = model(X, return_all=True)
        
        recon_loss = 0
        for i, (x_recon, _) in enumerate(all_layer_outputs):
            if i == 0:
                original_input = X
            else:
                _, prev_hidden = all_layer_outputs[i-1]
                original_input = prev_hidden.detach()
            
            # 确保张量在相同设备
            recon_loss += F.mse_loss(x_recon, original_input.to(device))
        
        # 稀疏损失
        _, last_hidden = all_layer_outputs[-1]
        sparse_loss = model.layers[-1].sparse_loss(last_hidden)
        
        total_loss = recon_loss + model.layers[-1].sparsity_weight * sparse_loss
        total_loss.backward()
        optimizer.step()
        swanlab.log({"fine-tune-loss": total_loss.item()})
        
        if (epoch+1) % 10 == 0:
            print(f"Fine-tuning Epoch {epoch+1}/{epochs} Loss: {total_loss.item():.4f}")
    
    return final_emb.cpu().detach()  # 返回CPU端数据用于评估


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

    print(f"ACC: {acc:.4f}")
    print(f"Adjusted Rand Index: {ari_score:.4f}")
    print(f"Normalized Mutual Information: {nmi_score:.4f}")
    print(f"Homogeneity: {homogeneity:.4f}")
    print(f"Completeness: {completeness:.4f}")
    print(f"V-Measure: {v_measure:.4f}")
    return acc, ari_score, nmi_score, homogeneity, completeness, v_measure
