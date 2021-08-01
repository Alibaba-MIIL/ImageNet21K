import torch

from src_files.helper_functions.distributed import reduce_tensor, num_distrib


class AccuracySemanticSoftmaxMet:
    "Average the values of `func` taking into account potential different batch sizes"

    def __init__(self, semantic_softmax_processor):
        self.semantic_softmax_processor = semantic_softmax_processor
        self.total, self.count = 0., 0

    def reset(self):
        self.total, self.count = 0., 0

    def accumulate(self, logits, targs):
        with torch.no_grad():
            semantic_logit_list = self.semantic_softmax_processor.split_logits_to_semantic_logits(logits)
            semantic_targets_tensor = self.semantic_softmax_processor.convert_targets_to_semantic_targets(targs)
            accuracy_list = []
            accuracy_valid_list = []
            result = 0
            for i in range(len(semantic_logit_list)):  # scanning hirarchy_level_list
                logits_i = semantic_logit_list[i]
                targets_i = semantic_targets_tensor[:, i]
                pred_i = logits_i.argmax(dim=-1)
                ind_valid = (targets_i >= 0)
                num_valids = torch.sum(ind_valid)
                accuracy_valid_list.append(num_valids)
                if num_valids > 0:
                    accuracy_list.append((pred_i[ind_valid] == targets_i[ind_valid]).float().mean())
                else:
                    accuracy_list.append(0)
                result += accuracy_list[-1] * accuracy_valid_list[-1]
            num_valids_total = sum(accuracy_valid_list)

        result = result.detach()
        num_valids_total = num_valids_total.detach()
        num_valids_total = num_valids_total.float()
        if num_distrib() > 1:
            result = reduce_tensor(result, num_distrib())
            num_valids_total = reduce_tensor(num_valids_total, num_distrib())

        self.total += result.item()
        self.count += num_valids_total.item()

    @property
    def value(self):
        return self.total / self.count * 100 if self.count != 0 else None

    @property
    def name(self):
        return "semantic_accuracy"
