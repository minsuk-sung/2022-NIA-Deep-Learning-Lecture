import os
import torch
from tensorboardX import SummaryWriter

use_cuda = torch.cuda.is_available()

default_checkpoint = {
    "epoch": 0,

    # train
    "train_losses": [],
    "train_accuracy": [],
    "train_precision": [],
    "train_recall": [],

    # valid
    "valid_losses": [],
    "valid_accuracy": [],
    "valid_precision": [],
    "valid_recall": [],

    "lr": [], 
    "model": {},
    "configs":{},
}

def save_checkpoint(checkpoint, dir="./checkpoints", prefix=""):
    filename = "{num:0>4}.pth".format(num=checkpoint["epoch"])
    if not os.path.exists(os.path.join(prefix, dir)):
        os.makedirs(os.path.join(prefix, dir))
    torch.save(checkpoint, os.path.join(prefix, dir, filename))

def load_checkpoint(path, cuda=use_cuda):
    if cuda:
        return torch.load(path)
    else:
        # Load GPU model on CPU
        return torch.load(path, map_location=lambda storage, loc: storage)

def init_tensorboard(name="", base_dir="./tensorboard"):
    return SummaryWriter(os.path.join(name, base_dir))

def write_tensorboard(
    writer,
    epoch,
    train_loss,
    train_accuracy,
    train_precision,
    train_recall,
    valid_loss,
    valid_accuracy,
    valid_precision,
    valid_recall,
    model,
):
    writer.add_scalar("train_loss", train_loss, epoch)
    writer.add_scalar("train_accuracy", train_accuracy, epoch)
    writer.add_scalar("train_precision", train_precision, epoch)
    writer.add_scalar("train_recall", train_recall, epoch)
    writer.add_scalar("valid_loss", valid_loss, epoch)
    writer.add_scalar("valid_accuracy", valid_accuracy, epoch)
    writer.add_scalar("valid_precision", valid_precision, epoch)
    writer.add_scalar("valid_recall", valid_recall, epoch)

    for name, param in model.named_parameters():
        writer.add_histogram(
            "{}".format(name), param.detach().cpu().numpy(), epoch
        )
        if param.grad is not None:
            writer.add_histogram(
                "{}/grad".format(name), param.grad.detach().cpu().numpy(), epoch
                )