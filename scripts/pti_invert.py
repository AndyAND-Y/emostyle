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

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')

    # Get a list of all items in the directory
    all_items = os.listdir(datapath)

    # Filter for files that have one of the image extensions
    image_files = [
        item for item in all_items
        if os.path.isfile(os.path.join(datapath, item)) and item.lower().endswith(image_extensions)
    ]

    for image_name in tqdm(image_files):

        image_path = os.path.join(datapath, image_name)

        # print(f'inverting, {image_path}')
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        latent = inversion.invert(
            image, os.path.basename(image_path).split('.')[0])


if __name__ == "__main__":
    invert(
        datapath="D:\\ML\\data\\upscaled\\ordered-256x256",
        inversion_type="e4e"  # e4e (wplus), w_encoder (w)
    )
