import multiprocessing
import os
from torch.utils.data import Dataset, DataLoader
from utils.imageDataset import ImageDatasetAFF
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image


def process_image(args):

    img_path, folder_path, output_folder_path = args

    image = Image.open(img_path).convert("RGB")

    upscaled_image = image.resize((1024, 1024), Image.Resampling.BICUBIC)

    relative_path = os.path.relpath(img_path, folder_path)

    output_path = os.path.join(
        output_folder_path, relative_path).replace("\\", "/")

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    upscaled_image.save(output_path)

    return True


def upscale_folder(folder_path, output_folder_path):

    upscale_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((1024, 1024), Image.Resampling.LANCZOS),
    ])

    all_image_paths_dataset = ImageDatasetAFF(
        folder_path, transform=upscale_transform, load_every_n=10
    )
    all_image_paths = all_image_paths_dataset.image_paths

    task_args = [(img_path, folder_path, output_folder_path)
                 for img_path in all_image_paths]

    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(process_image, task_args),
                total=len(task_args),
                desc="Upscaling Images"
            )
        )


def main():

    folder_path = "D:/ML/data/cropped-aligned"
    output_folder_path = "D:/ML/data/upscaled"
    upscale_folder(folder_path, output_folder_path)


if __name__ == '__main__':
    main()
