import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

#https://towardsdatascience.com/a-beginners-guide-to-graph-neural-networks-using-pytorch-geometric-part-1-d98dc93e7742

class Net(torch.nn.Module):
    def __init__(self):
        # define layers
        
        super(Net, self).__init__()
        #self.conv1 = GCNConv(data.num_features, 16)
        #self.conv2 = GCNConv(16, int(data.num_classes))
        
    def forward(self):
        # define data flow
        
        return self