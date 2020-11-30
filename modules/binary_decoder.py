import torch.nn as nn
import torch

class BinaryDecoder(nn.Module):
    def __init__(self, hidden_dim, num_label):
        super(BinaryDecoder, self).__init__()
        self.num_label = num_label
        self.binary_hidden2label_list = nn.ModuleList([nn.Linear(hidden_dim, 2) for _ in range(num_label)])

    def forward(self, out):
        pred_list = [self.binary_hidden2label_list[i](out) for i in range(self.num_label)]
        return torch.stack(pred_list, dim=1)
