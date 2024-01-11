import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    total_rmse = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, depth_true = batch['image'], batch['depth']
            depth_true = depth_true.unsqueeze(1)

            # move images and depth maps to correct device and type
            image = image.to(device=device, dtype=torch.float32)
            depth_true = depth_true.to(device=device, dtype=torch.float32)

            # predict the depth
            depth_pred = net(image)

            # Compute RMSE
            rmse = torch.sqrt(torch.mean((depth_pred - depth_true) ** 2))
            total_rmse += rmse

    net.train()
    return total_rmse / max(num_val_batches, 1)