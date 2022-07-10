import torch
import torch.functional

class ContrastiveLoss(torch.nn.Module):

    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.cos_sim = torch.nn.CosineSimilarity()

    def forward(self, inputs, positives, negatives):
        batch_size = inputs.shape[0]
        min_same_pair = self.cos_sim(inputs, positives).abs().mean()
        max_diff_pair = self.cos_sim(inputs, negatives).abs().mean()

        return F.relu(min_same_pair + max_diff_pair)
