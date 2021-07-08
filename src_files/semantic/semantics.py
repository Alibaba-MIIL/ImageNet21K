import torch
import numpy as np
from torch import Tensor


@torch.jit.script
def stable_softmax(logits: torch.Tensor):
    logits_m = logits - logits.max(dim=1)[0].unsqueeze(1)
    exp = torch.exp(logits_m)
    probs = exp / torch.sum(exp, dim=1).unsqueeze(1)
    return probs


class ImageNet21kSemanticSoftmax:
    def __init__(self, args):
        self.args = args
        self.tree = torch.load(args.tree_path)
        self.class_tree_list = self.tree['class_tree_list']
        self.class_names = np.array(list(self.tree['class_description'].values()))
        self.max_normalization_factor = 2e1
        num_classes = len(self.class_tree_list)
        self.class_depth = torch.zeros(num_classes)
        for i in range(num_classes):
            self.class_depth[i] = len(self.class_tree_list[i]) - 1
        max_depth = int(torch.max(self.class_depth).item())

        # process semantic relations
        hist_tree = torch.histc(self.class_depth, bins=max_depth + 1, min=0, max=max_depth).int()
        ind_list = []
        class_names_ind_list = []
        hirarchy_level_list = []
        cls = torch.tensor(np.arange(num_classes))
        for i in range(max_depth):
            if hist_tree[i] > 1:
                hirarchy_level_list.append(i)
                ind_list.append(cls[self.class_depth == i].long())
                class_names_ind_list.append(self.class_names[ind_list[-1]])
        self.hierarchy_indices_list = ind_list
        self.hirarchy_level_list = hirarchy_level_list
        self.class_names_ind_list = class_names_ind_list

        # calcuilating normalization array
        self.normalization_factor_list = torch.zeros_like(hist_tree)
        self.normalization_factor_list[-1] = hist_tree[-1]
        for i in range(max_depth):
            self.normalization_factor_list[i] = torch.sum(hist_tree[i:], dim=0)
        self.normalization_factor_list = self.normalization_factor_list[0] / self.normalization_factor_list
        if self.max_normalization_factor:
            self.normalization_factor_list.clamp_(max=self.max_normalization_factor)

    def split_logits_to_semantic_logits(self, logits: Tensor) -> Tensor:
        """
        split logits to 11 different hierarchies.

        :param self.self.hierarchy_indices_list: a list of size [num_of_hierarchies].
        Each element in the list is a tensor that contains the corresponding indices for the relevant hierarchy
        """
        semantic_logit_list = []
        for i, ind in enumerate(self.hierarchy_indices_list):
            logits_i = logits[:, ind]
            semantic_logit_list.append(logits_i)
        return semantic_logit_list

    def convert_targets_to_semantic_targets(self, targets_original: Tensor) -> Tensor:
        """
        converts single-label targets to targets over num_of_hierarchies different hierarchies.
        [batch_size] -> [batch_size x num_of_hierarchies].
        if no hierarchical target is available, outputs -1.

        :param self.self.hierarchy_indices_list: a list of size [num_of_hierarchies].
        Each element in the list is a tensor that contains the corresponding indices for the relevant hierarchy

        :param self.class_tree_list: a list of size [num_of_classes].
        Each element in the list is a list of the relevent sub-hirrachies.
        example - self.class_tree_list[10]:  [10, 9, 66, 65, 144]

        """
        targets = targets_original.cpu().detach()  # dont edit original targets
        semantic_targets_list = torch.zeros((targets.shape[0], len(self.hierarchy_indices_list))) - 1
        for i, target in enumerate(targets.cpu()):  # scanning over batch size
            cls_multi_list = self.class_tree_list[target]  # all the sub-hirrachies of the rager
            hir_levels = len(cls_multi_list)
            for j, cls in enumerate(cls_multi_list):
                # protection for too small hirarchy_level_list. this protection enables us to remove hierarchies
                if len(self.hierarchy_indices_list) <= hir_levels - j - 1:
                    continue
                ind_valid = (self.hierarchy_indices_list[hir_levels - j - 1] == cls)
                semantic_targets_list[i, hir_levels - j - 1] = torch.where(ind_valid)[0]

        return semantic_targets_list.long().to(device=targets_original.device)

    def estimate_teacher_confidence(self, preds_teacher: Tensor) -> Tensor:
        """
        Helper function:
        return the sum probabilities of the top 5% classes in preds_teacher.
        preds_teacher dimensions - [batch_size x num_of_classes]
        """
        with torch.no_grad():
            num_elements = preds_teacher.shape[1]
            num_elements_topk = int(np.ceil(num_elements / 20))  # top 5%
            weights_batch = torch.sum(torch.topk(preds_teacher, num_elements_topk).values, dim=1)
        return weights_batch

    def calculate_KD_loss(self, input_student: Tensor, input_teacher: Tensor):
        """
        Calculates the semantic KD-MSE distance between student and teacher probabilities
        input_student dimensions - [batch_size x num_of_classes]
        input_teacher dimensions - [batch_size x num_of_classes]
        """

        semantic_input_student = self.split_logits_to_semantic_logits(input_student)
        semantic_input_teacher = self.split_logits_to_semantic_logits(input_teacher)
        number_of_hierarchies = len(semantic_input_student)

        losses_list = []
        # scanning hirarchy_level_list
        for i in range(number_of_hierarchies):
            # converting to semantic logits
            inputs_student_i = semantic_input_student[i]
            inputs_teacher_i = semantic_input_teacher[i]

            # generating probs
            preds_student_i = stable_softmax(inputs_student_i)
            preds_teacher_i = stable_softmax(inputs_teacher_i)

            # weight MSE-KD distances according to teacher confidence
            loss_non_reduced = torch.nn.MSELoss(reduction='none')(preds_student_i, preds_teacher_i)
            weights_batch = self.estimate_teacher_confidence(preds_teacher_i)
            loss_weighted = loss_non_reduced * weights_batch.unsqueeze(1)
            losses_list.append(torch.sum(loss_weighted))

        return sum(losses_list)
