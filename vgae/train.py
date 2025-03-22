import torch
from torch_geometric.datasets import Planetoid

from model import VGAEModel, kmeans, eval

channels = 16
lr = 0.01
epochs = 200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid(root='../dataset', name='Cora')
data = dataset[0]
model = VGAEModel(dataset.num_features, channels).to(device)
data = data.to(device)

model.pretrain(data, epochs, lr)
embeddings = model.evaluate(data).cpu().numpy()
cluster_labels = kmeans(embeddings, dataset.num_classes)
eval(cluster_labels, data.y.cpu().numpy())  