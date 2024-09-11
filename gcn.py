!pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric

!pip install torch-geometric

# importing libs
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]

def accuracy(y_pred, y_true):
  return torch.sum(y_pred == y_true) / len(y_true)

# graph convolutional network
class GCN(torch.nn.Module):
  def __init__(self, dim_in, dim_h, dim_out):
    super().__init__()
    self.gcn1 = GCNConv(dim_in, dim_h)
    self.gcn2 = GCNConv(dim_h, dim_out)

  def forward(self, x, edge_index):
    h = self.gcn1(x, edge_index)
    h = torch.relu(h)
    h = self.gcn2(h, edge_index)
    return F.log_softmax(h, dim=1)

  def fit(self, data, epochs):
    optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    self.train()
    for epoch in range(epochs+1):
      optimizer.zero_grad()
      out = self(data.x, data.edge_index)
      loss = criterion(out[data.train_mask], data.y[data.train_mask])
      acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
      loss.backward()
      optimizer.step()

      if(epoch % 20 == 0):
        val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
        val_acc = accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])
        print(f'Epoch: {epoch:>3}, Train Loss: {loss:.3f}, Train Acc: {acc*100:>5.2f}%, Val Loss: {val_loss:.2f}, Val Acc: {val_acc*100:.2f}%')

  @torch.no_grad()
  def test(self, data):
    self.eval()
    out = self(data.x, data.edge_index)
    acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
    return acc

# creating the GCN model
gcn = GCN(dataset.num_features, 16, dataset.num_classes)
print(gcn)

# train
gcn.fit(data, epochs=100)

# evaluate
acc = gcn.test(data)
print(f'\nGCN test Accuracy: {acc*100:.2f}%\n')













