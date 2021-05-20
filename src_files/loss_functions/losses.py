import torch
import torch.nn as nn

class CrossEntropyLS(nn.Module):
    def __init__(self, eps: float = 0.2):
        super(CrossEntropyLS, self).__init__()
        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, target):
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)
        targets_classes.mul_(1 - self.eps).add_(self.eps / num_classes)
        cross_entropy_loss_tot = -targets_classes.mul(log_preds)
        cross_entropy_loss = cross_entropy_loss_tot.sum(dim=-1).mean()
        return cross_entropy_loss
