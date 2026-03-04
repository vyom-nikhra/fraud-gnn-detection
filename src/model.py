import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class FraudSAGE(nn.Module):
    def __init__(self, num_node_features: int, hidden_channels: int, num_classes: int):
        super(FraudSAGE, self).__init__()
        self.conv1 = SAGEConv(num_node_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x