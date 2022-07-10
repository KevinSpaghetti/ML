import torch

class ContrastiveLoss(torch.nn.Module):

    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, inputs, targets):
        batch_size = inputs.shape[0]
        return torch.cdist(inputs, targets, p=2)
