import os
import queue
import threading
import numpy as np
import torch
from utils.imageDataset import ImageDatasetAFF
from torchvision import transforms
from torch.utils.data import DataLoader
from models.e4e.psp import pSp
from argparse import Namespace
from tqdm import tqdm
from torchvision.utils import save_image

QUEUE_MAX_SIZE = 300
SAVING_WORKERS = os.cpu_count()
DATALOADER_WORKERS = 4


def save_worker(q):
    while True:
        item = q.get(block=True)
        if item is None:
            break

        original_path, latent_code, generated_image_tensor = item

        try:
            dirname = os.path.dirname(original_path)
            dirname = os.path.join(dirname, 'inverted')

            filename_without_ext = os.path.splitext(
                os.path.basename(original_path)
            )[0]

            output_path_latent = os.path.join(
                dirname, filename_without_ext + '.npy')
            output_path_image = os.path.join(
                dirname, filename_without_ext + '.jpg')

            os.makedirs(dirname, exist_ok=True)

            np.save(output_path_latent, latent_code)

            save_image(
                generated_image_tensor,
                output_path_image, normalize=True,
            )

        except Exception as e:
            print(f"Error saving files for {original_path}: {e}")


def mass_invert(folder_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path_e4e = 'pretrained/e4e_ffhq_encode.pt'

    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    datatset = ImageDatasetAFF(folder_path, preprocess)

    loader = DataLoader(
        datatset,
        batch_size=4,
        shuffle=False,
        pin_memory=True,
        num_workers=DATALOADER_WORKERS
    )

    ckpt = torch.load(
        path_e4e, map_location='cpu', weights_only=True)

    opts = ckpt['opts']
    opts['device'] = device
    opts['checkpoint_path'] = path_e4e
    opts = Namespace(**opts)

    e4e_model = pSp(opts)\
        .eval()\
        .to(device)\
        .requires_grad_(False)

    e4e_model = torch.compile(e4e_model, backend='cudagraphs')

    save_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
    save_workers = []
    print(f"Starting {SAVING_WORKERS} saving workers...")
    for _ in range(SAVING_WORKERS):
        worker = threading.Thread(target=save_worker, args=(save_queue,))
        worker.start()
        save_workers.append(worker)

    for images, paths in tqdm(loader, desc="Inverting images..."):

        images = images.to(device)

        with torch.no_grad():
            outputs = e4e_model(
                images,
                randomize_noise=False,
                return_latents=True,
                resize=False,
                input_code=False,
            )

            for i, path in enumerate(paths):
                original_path = path
                latent_code = outputs[1][i].cpu().numpy()
                generated_image_tensor = outputs[0][i].cpu()

                save_queue.put(
                    (original_path, latent_code, generated_image_tensor)
                )

    print("Finished processing batch. Signaling saving workers to stop...")
    for _ in range(SAVING_WORKERS):
        save_queue.put(None)

    print("Waiting for saving workers to complete...")
    for worker in save_workers:
        worker.join()

    print("Mass inversion complete.")


def main():

    folder_path = 'D:\\ML\\data\\upscaled\\ordered-1024x1024'
    mass_invert(folder_path)


if __name__ == '__main__':
    main()
