import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import DepthDataset
from unet import UNet
from utils.utils import plot_img_and_mask

def predict_depth(net, full_img, device, scale_factor=0.5):
    net.eval()
    img = torch.from_numpy(DepthDataset.preprocess(full_img, scale_factor, is_depth=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = torch.nn.functional.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        depth = output.squeeze().numpy()

    return depth


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='/home/cip/2022/ce90tate/UNet_CompVP/checkpoints/checkpoint_epoch10.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return list(map(_generate_name, args))



def depth_to_image(depth: np.ndarray):
    # Normalize the depth for visualization
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth = (depth * 255).astype(np.uint8)
    return Image.fromarray(depth)


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input[0].split(",")
    out_files = get_output_filenames(in_files)

    net = UNet(n_channels=1, n_classes=1, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting depth for image {filename} ...')
        img = Image.open(filename.strip())
        img = img.convert('L')

        depth = predict_depth(net=net,
                              full_img=img,
                              scale_factor=args.scale,
                              device=device)

        if not args.no_save:
            out_filename = out_files[i]
            result = depth_to_image(depth)
            result.save(out_filename.strip())
            logging.info(f'Depth map saved to {out_filename}')