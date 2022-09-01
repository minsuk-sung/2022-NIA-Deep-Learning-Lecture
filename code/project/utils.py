import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from tqdm import tqdm

def get_network(
    options
):
    model = None

    if options.network == "VGG":        
        model = models.vgg11_bn(pretrained=options.model.pretrained)
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, options.data.num_classes),
        )
    else:
        raise NotImplementedError

    return model.to(options.device)

def get_optimizer(
    params,
    options
):
    if options.optimizer.type == "Adam":
        optimizer = optim.Adam(params, lr=options.optimizer.lr, weight_decay=options.optimizer.weight_decay)
    else:
        raise NotImplementedError

    return optimizer
