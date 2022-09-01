'''
USAGE
python train.py --config_file=configs/ResNet.yaml
'''

import os
import time
import fire
import yaml
import random
import torch
import shutil
import argparse
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from psutil import virtual_memory

from flags import Flags
from checkpoint import (
    default_checkpoint,
    load_checkpoint,
    save_checkpoint,
    init_tensorboard,
    write_tensorboard,
)
from utils import get_network, get_optimizer
from dataset import get_train_valid_dataloader
from metrics import accuracy, precision, recall


def run_epoch(
    options,
    data_loader,
    model,
    epoch_text,
    optimizer,
    lr_scheduler,
    train=True,
):
    torch.set_grad_enabled(train)
    if train:
        model.train()
    else:
        model.eval()

    losses = []
    acces = []
    precisions = []
    recalls = []

    with tqdm(
        desc="{} ({})".format(epoch_text, "Train" if train else "Valid"),
        total=len(data_loader.dataset),
        dynamic_ncols=True,
        leave=False,
    ) as pbar:
        for i, (images, targets) in enumerate(data_loader):
            images = images.to(options.device, torch.float)
            targets = targets.to(options.device, torch.long)

            curr_batch_size = len(images)

            scores = model(images).to(options.device)
            _, preds = scores.max(dim=1)

            loss = F.cross_entropy(scores, targets)
            acc = accuracy(targets, preds, options.batch_size)
            pre = precision(targets, preds)
            rec = recall(targets, preds)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            losses.append(loss.item())
            acces.append(acc)
            precisions.append(pre)
            recalls.append(rec)

            pbar.update(curr_batch_size)

    lr_scheduler.step()

    result = {
        "loss": np.mean(losses),
        "accuracy": np.mean(acces),
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
    }

    return result

def main(config_file):
    """
    Train math formula recognition model
    """
    options = Flags(config_file).get()

    random.seed(options.seed)
    np.random.seed(options.seed)
    os.environ["PYTHONHASHSEED"] = str(options.seed)
    torch.manual_seed(options.seed)
    torch.cuda.manual_seed(options.seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

    is_cuda = torch.cuda.is_available()
    print("--------------------------------")
    print("Running {} on device {}\n".format(options.network, options.device))

    current_device = torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'
    num_gpus = torch.cuda.device_count()
    num_cpus = os.cpu_count()
    mem_size = virtual_memory().available // (1024 ** 3)
    torch.cuda.empty_cache()
    print(
        "[+] System environments\n",
        "Device: {}\n".format(torch.cuda.get_device_name(current_device)),
        "Random seed : {}\n".format(options.seed),
        "The number of gpus : {}\n".format(num_gpus),
        "The number of cpus : {}\n".format(num_cpus),
        "Memory Size : {}G\n".format(mem_size),
    )

    checkpoint = (
        load_checkpoint(options.checkpoint, cuda=is_cuda)
        if options.checkpoint != ""
        else default_checkpoint
    )

    train_data_loader, valid_data_loader, train_dataset, valid_dataset = get_train_valid_dataloader(options)
    print(
        "[+] Data\n",
        "Train path : {}\n".format(options.data.train),
        "Test path : {}\n".format(options.data.test),
        "Batch size : {}\n".format(options.batch_size),
        "Valid proportions : {}\n".format(options.data.test_proportions),
        "The number of train samples : {:,}\n".format(len(train_dataset)),
        "The number of valid samples : {:,}\n".format(len(valid_dataset)),
    )

    model = get_network(options)
    model_state = checkpoint.get("model")
    if model_state:
        model.load_state_dict(model_state)
        print(
        "[+] Checkpoint\n",
        "Resuming from epoch : {}\n".format(checkpoint["epoch"]),
        "Train Accuracy : {:.5f}\n".format(checkpoint["train_accuracy"][-1]),
        "Train Loss : {:.5f}\n".format(checkpoint["train_losses"][-1]),
        "Valid Accuracy : {:.5f}\n".format(checkpoint["valid_accuracy"][-1]),
        "Valid Loss : {:.5f}\n".format(checkpoint["valid_losses"][-1]),
        )
    
    params_to_optimise = [
        param for param in model.parameters() if param.requires_grad
    ]
    print(
        "[+] Network\n",
        "Type: {}\n".format(options.network),
        "Model parameters: {:,}\n".format(
            sum(p.numel() for p in params_to_optimise),
        ),
    )

    optimizer = get_optimizer(params_to_optimise, options)
    optimizer_state = checkpoint.get("optimizer")
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)
    for param_group in optimizer.param_groups:
        param_group["initial_lr"] = options.optimizer.lr

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    print(
        "[+] Optimizer\n",
        "Type: {}\n".format(options.optimizer.type),
        "Learning rate: {:,}\n".format(options.optimizer.lr),
        "Weight Decay: {:,}\n".format(options.optimizer.weight_decay),
    )

    if not os.path.exists(options.prefix):
        os.makedirs(options.prefix)
    log_file = open(os.path.join(options.prefix, "log.txt"), "w")
    shutil.copy(config_file, os.path.join(options.prefix, "train_config.yaml"))

    if options.print_epochs is None:
        options.print_epochs = options.num_epochs
        
    writer = init_tensorboard(name=options.prefix.strip("-"))
    start_epoch = checkpoint["epoch"]
    train_accuracy = checkpoint["train_accuracy"]
    train_recall = checkpoint["train_recall"]
    train_precision = checkpoint["train_precision"]
    train_losses = checkpoint["train_losses"]
    valid_accuracy = checkpoint["valid_accuracy"]
    valid_recall = checkpoint["valid_recall"]
    valid_precision = checkpoint["valid_precision"]
    valid_losses = checkpoint["valid_losses"]
    learning_rates = checkpoint["lr"]

    valid_early_stop = 0
    valid_best_loss = float('inf')

    for epoch in range(options.num_epochs):
        start_time = time.time()

        epoch_text = "[{current:>{pad}}/{end}] Epoch {epoch}".format(
            current=start_epoch + epoch + 1,
            end=start_epoch + options.num_epochs,
            epoch=start_epoch + epoch + 1,
            pad=len(str(options.num_epochs)),
        )

        train_result = run_epoch(
            options,
            train_data_loader,
            model,
            epoch_text,
            optimizer,
            lr_scheduler,
            train=True,
        )

        train_losses.append(train_result["loss"])
        train_precision.append(train_result["precision"])
        train_recall.append(train_result["recall"])
        train_accuracy.append(train_result["accuracy"])

        epoch_lr = lr_scheduler.get_last_lr()[-1]

        valid_result = run_epoch(
            options,
            valid_data_loader,
            model,
            epoch_text,
            optimizer,
            lr_scheduler,
            train=False,
        )

        valid_losses.append(valid_result["loss"])
        valid_precision.append(valid_result["precision"])
        valid_recall.append(valid_result["recall"])
        valid_accuracy.append(valid_result["accuracy"])

        with open(config_file, 'r') as f:
            option_dict = yaml.safe_load(f)

        save_checkpoint(
            {
                "epoch": start_epoch + epoch + 1,
                "train_losses": train_losses,
                "train_accuracy": train_accuracy,
                "train_precision": train_precision,
                "train_recall": train_recall,
                "valid_losses": valid_losses,
                "valid_accuracy":valid_accuracy,
                "valid_precision": valid_precision,
                "valid_recall": valid_recall,
                "lr": learning_rates,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "configs": option_dict,
            },
            prefix=options.prefix,
        )

        elapsed_time = time.time() - start_time
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        if epoch % options.print_epochs == 0 or epoch == options.num_epochs - 1:
            output_string = (
                "{epoch_text}: "
                "Train Accuracy = {train_accuracy:.5f}, "
                "Train Precision = {train_precision:.5f}, "
                "Train Recall = {train_recall:.5f}, "
                "Train Loss = {train_loss:.5f}, "
                "Valid Accuracy = {valid_accuracy:.5f}, "
                "Valid Precision = {valid_precision:.5f}, "
                "Valid Recall = {valid_recall:.5f}, "
                "Valid Loss = {valid_loss:.5f}, "
                "lr = {lr} "
                "(time elapsed {time})"
            ).format(
                epoch_text=epoch_text,
                train_accuracy=train_result["accuracy"],
                train_precision=train_result["precision"],
                train_recall=train_result["recall"],
                train_loss=train_result["loss"],
                valid_accuracy=valid_result["accuracy"],
                valid_precision=valid_result["precision"],
                valid_recall=valid_result["recall"],
                valid_loss=valid_result["loss"],
                lr=epoch_lr,
                time=elapsed_time,
            )
            print(output_string)
            log_file.write(output_string + "\n")
            write_tensorboard(
                writer,
                start_epoch + epoch + 1,
                train_result["loss"],
                train_result["accuracy"],
                train_result["precision"],
                train_result["recall"],
                valid_result["loss"],
                valid_result["accuracy"],
                valid_result["precision"],
                valid_result["recall"],
                model,
            )

        if valid_result["loss"] < valid_best_loss:
            valid_best_loss = valid_result["loss"]
            valid_early_stop = 0

        else:
            valid_early_stop += 1
            if valid_early_stop >= options.EARLY_STOPPING_EPOCH:
                print("EARLY STOPPING!!")
                break

        

if __name__ == "__main__":
    fire.Fire(main)