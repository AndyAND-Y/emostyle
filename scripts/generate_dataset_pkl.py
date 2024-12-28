import os
import pickle
from typing import List

import numpy as np
import PIL.Image
import torch
from torch.nn.functional import batch_norm
import dnnlib
from tqdm import tqdm


def generate_data(
    checkpoint_path: str,
    stylegan_size: int,
    truncation_psi: float,
    n_samples: int,
    outdir: str,
):
    device = 'cuda'

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}"
        )

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    with dnnlib.util.open_url(checkpoint_path) as f:
        G = pickle.load(f)['G_ema'].to(device).eval().float()

    mapping = G.mapping.requires_grad_(False)
    synthesis = G.synthesis.requires_grad_(False)

    image_id = 0
    batch_size = 8

    for i in tqdm(range(n_samples // batch_size + 1), desc="Generating Images"):

        z = np.random.randn(batch_size, 512).astype("float32")

        with torch.no_grad():
            latents = mapping(
                torch.from_numpy(z).to(device),
                None,
                truncation_psi=truncation_psi
            )

            all_images = synthesis(
                latents,
                noise_mode='const',
                force_fp32=True
            )

            all_images = (all_images.permute(0, 2, 3, 1) * 127.5 + 128)\
                .clamp(0, 255).to(torch.uint8).cpu().numpy()

            for id in range(batch_size):

                img_path = f'{outdir}/{image_id:06}.png'
                npy_path = f'{outdir}/{image_id:06}.npy'

                PIL.Image.fromarray(
                    all_images[id],
                    'RGB'
                )\
                    .save(img_path)

                np.save(
                    npy_path,
                    latents[id].cpu().numpy(),
                    allow_pickle=False
                )

                image_id += 1


if __name__ == "__main__":
    generate_data("pretrained/ffhq2.pkl", 1024, 0.7, 2048, "dataset/1024_pkl")
