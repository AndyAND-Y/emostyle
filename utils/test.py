
from matplotlib import pyplot as plt
import numpy as np
from torchvision import transforms


import torch
from utils.imageDataset import ImageDataset
from torch.utils.data import DataLoader
from models.arcface import ArcFace

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
arcface_path = "pretrained/model_ir_se50.pth"
dataset_path = "dataset/1024_pkl/"


def get_id():

    arcface = ArcFace(arcface_path)
    arcface.eval()

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    original_images_loader = DataLoader(
        ImageDataset(dataset_path, transform=transform),
        shuffle=False
    )

    img1 = ImageDataset(dataset_path, transform=transform)[0]
    img2 = ImageDataset(dataset_path, transform=transform)[0]

    plt.imshow(img1.squeeze(0), cmap='gray')
    plt.axis("off")
    plt.title(f"Image Index: {0}")
    plt.show()

    plt.imshow(img2.squeeze(0), cmap='gray')
    plt.axis("off")
    plt.title(f"Image Index: {1}")
    plt.show()

    id_score = torch.nn.functional.cosine_similarity(
        arcface(img1.unsqueeze(0)),
        arcface(img2.unsqueeze(0))
    )

    return id_score


if __name__ == '__main__':
    print(get_id())
