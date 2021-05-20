# --------------------------------------------------------
# ImageNet-21K Pretraining for The Masses
# Copyright 2021 Alibaba MIIL (c)
# Licensed under MIT License [see the LICENSE file for details]
# Written by Tal Ridnik
# --------------------------------------------------------

import argparse
import time
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
from torch.optim import lr_scheduler

from src_files.data_loading.data_loader import create_data_loaders
from src_files.helper_functions.distributed import print_at_master, to_ddp, num_distrib, setup_distrib
from src_files.helper_functions.general_helper_functions import silence_PIL_warnings
from src_files.models import create_model
from torch.cuda.amp import GradScaler, autocast
from src_files.optimizers.create_optimizer import create_optimizer
from src_files.semantic.metrics import AccuracySemanticSoftmaxMet
from src_files.semantic.semantic_loss import SemanticSoftmaxLoss
from src_files.semantic.semantics import ImageNet21kSemanticSoftmax

parser = argparse.ArgumentParser(description='PyTorch ImageNet21K Semantic Softmax Training')
parser.add_argument('--data_path', type=str)
parser.add_argument('--lr', default=3e-4, type=float)
parser.add_argument('--model_name', default='tresnet_m')
parser.add_argument('--model_path', default='./tresnet_m.pth', type=str)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--image_size', default=224, type=int)
parser.add_argument('--num_classes', default=11221, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--epochs', default=80, type=int)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--label_smooth", default=0.2, type=float)
parser.add_argument("--tree_path", default='./resources/imagenet21k_miil_tree.pth', type=str)


def main():
    # arguments
    args = parser.parse_args()

    # EXIF warning silent
    silence_PIL_warnings()

    # setup distributed
    setup_distrib(args)

    # Setup model
    model = create_model(args).cuda()
    model = to_ddp(model, args)

    # create optimizer
    optimizer = create_optimizer(model, args)

    # setup distributed
    model = to_ddp(model, args)

    # Data loading
    train_loader, val_loader = create_data_loaders(args)

    # semantic
    semantic_softmax_processor = ImageNet21kSemanticSoftmax(args)
    semantic_met = AccuracySemanticSoftmaxMet(semantic_softmax_processor)

    # Actuall Training
    train_21k(model, train_loader, val_loader, optimizer, semantic_softmax_processor, semantic_met, args)


def train_21k(model, train_loader, val_loader, optimizer, semantic_softmax_processor, met, args):
    # set loss
    loss_fn = SemanticSoftmaxLoss(semantic_softmax_processor)

    # set scheduler
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
                                        epochs=args.epochs, pct_start=0.1, cycle_momentum=False, div_factor=20)

    # set scalaer
    scaler = GradScaler()
    for epoch in range(args.epochs):
        if num_distrib() > 1:
            train_loader.sampler.set_epoch(epoch)

        # train epoch
        print_at_master("\nEpoch {}".format(epoch))
        epoch_start_time = time.time()
        for i, (input, target) in enumerate(train_loader):
            with autocast():  # mixed precision
                output = model(input)
                loss = loss_fn(output, target) # note - loss also in fp16
            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        epoch_time = time.time() - epoch_start_time
        print_at_master(
            "\nFinished Epoch, Training Rate: {:.1f} [img/sec]".format(len(train_loader) *
                                                                      args.batch_size / epoch_time * max(num_distrib(),
                                                                                                         1)))

        # validation epoch
        validate_21k(val_loader, model, met)


def validate_21k(val_loader, model, met):
    print_at_master("starting validation")
    model.eval()
    met.reset()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            # mixed precision
            with autocast():
                logits = model(input).float()

            # measure accuracy and record loss
            met.accumulate(logits, target)

    print_at_master("Validation results:")
    print_at_master('Semantic Acc_Top1 [%] {:.2f} '.format(met.value))
    model.train()


if __name__ == '__main__':
    main()
