import torch
from torch.optim import lr_scheduler

from src_files.helper_functions.general_helper_functions import add_weight_decay
from src_files.loss_functions.losses import CrossEntropyLS


def create_optimizer(model, args):
    parameters = add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=args.lr, weight_decay=0)  # true wd, filter_bias_and_bn
    return optimizer

def create_optimizer_sgd(model, args):
    parameters = add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.SGD(params=parameters, lr=args.lr, weight_decay=0)  # true wd, filter_bias_and_bn
    return optimizer

