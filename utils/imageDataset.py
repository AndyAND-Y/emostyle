import os
import random
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class ImageDatasetAFF(Dataset):
    def __init__(self, folder_path, transform=None, load_every_n=1):
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = []
        self.load_every_n = load_every_n
        for root, _, files in os.walk(folder_path):
            image_files_in_folder = []
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files_in_folder.append(os.path.join(
                        root, file.replace("\\", "/")
                    ))

            image_files_in_folder.sort()
            for i in range(0, len(image_files_in_folder), self.load_every_n):
                self.image_paths.append(image_files_in_folder[i])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, img_path
        except Exception as e:
            print(f"Error loading image: {img_path} - {e}")
            return None, img_path


class ImageDataset(Dataset):
    def __init__(self, image_dir, emotion_id=None, transform=None):
        self.image_dir = image_dir
        self.image_paths = []

        suffix = '.png'

        if emotion_id != None:
            suffix = f"-{emotion_id}" + suffix

        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.endswith(suffix):
                    self.image_paths.append(os.path.join(root, file))

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image


def plot_random_images(data_loader, num_images=10):
    images = []
    for _ in range(num_images):
        # Select a random batch and pick one image from that batch
        batch = next(iter(data_loader))
        image = random.choice(batch)
        images.append(image)

    # Plot the selected images
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i, image in enumerate(images):
        # Convert from CxHxW to HxWxC for plt
        axes[i].imshow(image.permute(1, 2, 0))
        axes[i].axis('off')

    plt.show()
