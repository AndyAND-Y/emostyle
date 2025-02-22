import os
import pickle
import random
import glob


import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import dnnlib

# from dataset import PersonalizedSyntheticDataset
from .invert import Inversion

is_cuda = torch.cuda.is_available()
EMO_EMBED = 64
STG_EMBED = 512
INPUT_SIZE = 1024


def invert(
    datapath: str,
    inversion_type: str
):
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    latent_path = os.path.join(datapath, 'latents/')
    if not os.path.exists(latent_path):
        os.makedirs(latent_path)
    inversion = Inversion(
        latent_path=os.path.join(latent_path),
        inversion_type=inversion_type,
        device='cuda' if is_cuda else 'cpu'
    )
    for image_name in tqdm(os.listdir(os.path.join(datapath, 'images/'))):

        image_path = f"{os.path.join(datapath, 'images/')}/{image_name}"

        print(f'inverting, {image_path}')
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        latent = inversion.invert(
            image, os.path.basename(image_path).split('.')[0])

        import pickle
        try:
            with dnnlib.util.open_url(str('pretrained/ffhq2.pkl')) as f:
                G = pickle.load(f)['G_ema'].synthesis
        except Exception as e:
            G = torch.load('pretrained/ffhq2.pkl')
        G = G.cuda()

        synthesis = G(latent, noise_mode='const', force_fp32=True).squeeze()

        image = (image.permute(1, 2, 0) * 127.5 + 128).clamp(0,
                                                             255).to(torch.uint8).cpu().numpy()
        synthesis = (synthesis.permute(1, 2, 0) * 127.5 +
                     128).clamp(0, 255).to(torch.uint8).cpu().numpy()

        Image.fromarray(image, 'RGB').save(
            f'experiments/results/{image_name.split('.')[0]}-original.{image_name.split('.')[1]}')
        Image.fromarray(synthesis, 'RGB').save(
            f'experiments/results/{image_name.split('.')[0]}-inverted.{image_name.split('.')[1]}')


if __name__ == "__main__":
    invert(
        datapath="experiments/",
        inversion_type="e4e"  # e4e (wplus), w_encoder (w)
    )
