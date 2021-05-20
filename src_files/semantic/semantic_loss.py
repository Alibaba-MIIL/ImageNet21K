import torch
import torch.nn.functional as F


class SemanticSoftmaxLoss(torch.nn.Module):
    def __init__(self, semantic_softmax_processor):
        super(SemanticSoftmaxLoss, self).__init__()
        self.semantic_softmax_processor = semantic_softmax_processor
        self.args = semantic_softmax_processor.args

    def forward(self, logits, targets):
        """
        Calculates the semantic cross-entropy loss distance between logits and targers
        """

        if not self.training:
            return 0

        semantic_logit_list = self.semantic_softmax_processor.split_logits_to_semantic_logits(logits)
        semantic_targets_tensor = self.semantic_softmax_processor.convert_targets_to_semantic_targets(targets)

        losses_list = []
        # scanning hirarchy_level_list
        for i in range(len(semantic_logit_list)):
            logits_i = semantic_logit_list[i]
            targets_i = semantic_targets_tensor[:, i]

            # generate probs
            log_preds = F.log_softmax(logits_i, dim=1)

            # generate targets (with protections)
            targets_i_valid = targets_i.clone()
            targets_i_valid[targets_i_valid < 0] = 0
            num_classes = logits_i.size()[-1]
            targets_classes = torch.zeros_like(logits_i).scatter_(1, targets_i_valid.unsqueeze(1), 1)
            targets_classes.mul_(1 - self.args.label_smooth).add_(self.args.label_smooth / num_classes)

            cross_entropy_loss_tot = -targets_classes.mul(log_preds)
            cross_entropy_loss_tot *= ((targets_i >= 0).unsqueeze(1))
            cross_entropy_loss = cross_entropy_loss_tot.sum(dim=-1)  # sum over classes
            loss_i = cross_entropy_loss.mean()  # mean over batch
            losses_list.append(loss_i)

        total_sum = 0
        for i, loss_h in enumerate(losses_list):  # summing over hirarchies
            total_sum += loss_h * self.semantic_softmax_processor.normalization_factor_list[i]

        return total_sum
