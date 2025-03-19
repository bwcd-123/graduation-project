import torch
from torch_geometric.datasets import Planetoid
import swanlab

from model import SelfOptimizingVGAE
from utils import predict, eval, train

channels = 512
lr = 0.001
epochs = 200
seed = torch.default_generator.seed()
# seed = 40473483797100 # no kaiming initial
# seed = 42739336062500 # with kaiming initial
print("current seed: ", seed)
dataset_name = 'Cora'
torch.manual_seed(seed)

swanlab.init(
    project='SelfOptimizingVGAE',
    description=f'SelfOptimizingVGAE on {dataset_name}',
    config={
        'channels': channels,
        'lr': lr,
        'epochs': epochs,
        'seed': seed,
        'dataset-name': dataset_name
    },
    log_dir='./logs'
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid(root='../dataset', name=dataset_name)
data = dataset[0]
model = SelfOptimizingVGAE(dataset.num_features, channels, dataset.num_classes).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
train(model, data, optimizer, epochs)
cluster_labels = predict(model, data)
ret = eval(data.y.cpu().numpy(), cluster_labels.cpu().numpy())
swanlab.log(ret)
swanlab.finish()
