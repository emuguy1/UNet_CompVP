import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch import optim
from tqdm import tqdm
from utils.data_loading import DepthDataset  # A custom dataset class for depth maps
from utils.metrics import depth_error_metric

dir_img = Path('data/training_set/cloth')
dir_mask = Path('data/training_set/distances')
dir_checkpoint = Path('./checkpoints/')


def custom_loss(output, target, input, dark_threshold=80, dark_weight=2.0):
    """
    Custom loss function that penalizes dark areas in the output where the input is also dark.
    :param output: Predicted depth map.
    :param target: Target depth map.
    :param input: Input image.
    :param dark_threshold: Threshold to classify pixels as 'dark' in the input.
    :param dark_weight: Additional weight for loss in dark areas.
    :return: Loss value.
    """
    mse_loss = F.mse_loss(output, target)  # Standard MSE loss

    # Identify dark regions in the input
    dark_regions = (input < dark_threshold).float()
    dark_region_loss = F.mse_loss(output * dark_regions, target * dark_regions)
    # Combine the standard loss with the dark region loss
    combined_loss = mse_loss + dark_weight * dark_region_loss
    return combined_loss

def get_latest_checkpoint(checkpoint_dir):
    """Get the latest checkpoint file from the given directory."""
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoint_files:
        return None
    latest_checkpoint = max(checkpoint_files, key=lambda f: os.path.getctime(os.path.join(checkpoint_dir, f)))
    return os.path.join(checkpoint_dir, latest_checkpoint)


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 10,
        learning_rate: float = 1e-2,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        gradient_clipping: float = 1.0,
        #weight_decay: float = 1e-8,
):
    # 1. Create dataset for depth estimation
    # Assuming DepthDataset is a custom class similar to BasicDataset but for depth maps
    dataset = DepthDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Begin training
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,  weight_decay=0.001)
    # Use Mean Squared Error Loss for depth estimation
    criterion = nn.MSELoss()

    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)



    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    global_step = 0
    model.to(device)

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, depth_maps = batch['image'], batch['depth']
                images = images.to(device)
                depth_maps = depth_maps.to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                depth_pred = model(images)
                loss = custom_loss(depth_pred, depth_maps, images)

                # Backward and optimize
                grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                epoch_loss += loss.item()

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(model, val_loader, device, amp)
                        logging.info('Validation score: {}'.format(val_score))

        if epoch % 10 == 0:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=50 , help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=15, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=1, n_classes=1)
    model = model.to(device=device)

    latest_checkpoint = get_latest_checkpoint(dir_checkpoint)
    if latest_checkpoint:
        logging.info(f'Loading model from latest checkpoint: {latest_checkpoint}')
        state_dict = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(state_dict)
    else:
        logging.info('No checkpoint found. Starting training from scratch.')

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )

