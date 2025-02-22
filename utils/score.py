"""
    Test config
    - specific emotions list (n*m emotions)
    - random emotion (1*1)
"""

"""
    Generate images
"""

"""
    test for va/std
    test for fid
    test for lp
    test for id 
"""

from torch.utils.data import  DataLoader
from utils.imageDataset import ImageDataset
import torch
import time
import random
import utils.inference
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
import numpy as np
import os
import lpips
import json


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def uint8_transform(x):
    return (x * 255).clamp(0, 255).to(torch.uint8)


def collate_fn(batch):
    return default_collate(batch).to(DEVICE)


def analyse_model_performance(output_dir: str, test_mode="random", isGenerated=False):

    stylegan2_path = "pretrained/ffhq2.pkl"
    emostyle_path = "checkpoints/emo_mapping_wplus_2.pt"
    dataset_path = "dataset/1024_pkl/"
    output_path = ""

    if not isGenerated:
        NUM_IMAGE = 1000
        valences = []
        arousals = []
        multiplier = 1
        output_path = f"{output_dir}/result-{time.strftime('%Y-%m-%d-%H-%M')}"
        os.makedirs(output_path, exist_ok=True)

        if (test_mode == "random"):
            valences = [random.uniform(-1, 1)]
            arousals = [random.uniform(-1, 1)]
            multiplier = 1

        if (test_mode == "list"):
            valences = [-1, -0.5, 0, 0.5, 1]
            arousals = [-1, -0.5, 0, 0.5, 1]
            multiplier = 25

        utils.inference.generate_edited_images(
            dataset_path,
            stylegan2_path,
            emostyle_path,
            output_path,
            valences,
            arousals,
            isRandom=test_mode == "random",
            limit=NUM_IMAGE//multiplier
        )
    else:
        output_path = output_dir

    def get_fid() -> float:

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(uint8_transform),
        ])

        fid_metric = FrechetInceptionDistance(
            input_img_size=(3, 1024, 1024)
        ).to(DEVICE)

        original_images_loader = DataLoader(
            ImageDataset(dataset_path, transform=transform),
            collate_fn=collate_fn
        )

        results = []

        for emotion_id in range(len(valences)*len(arousals)):

            emotion_edited_images_loader = DataLoader(
                ImageDataset(
                    output_path,
                    emotion_id=emotion_id,
                    transform=transform
                ),
                collate_fn=collate_fn
            )

            with torch.no_grad():

                for original_batch in original_images_loader:
                    fid_metric.update(original_batch, real=True)

                for edited_batch in emotion_edited_images_loader:
                    fid_metric.update(edited_batch, real=False)

                fid_score = fid_metric.compute()

                results.append(fid_score.item())

                fid_metric.reset()

        return sum(results)/len(results)

    def get_va():

        valences = []
        arousals = []

        for folder_name in os.listdir(output_path):
            folder_path = output_path + "/" + folder_name + "/"

            if os.path.isdir(folder_path) and folder_name.isdigit():
                json_file_path = os.path.join(
                    folder_path, f"result-{folder_name}.json")

                if os.path.exists(json_file_path):
                    with open(json_file_path, "r") as file:
                        data = json.load(file)
                        for edited_image in data.get("edited_images", []):
                            valences.append(edited_image["achieved_valence"])
                            arousals.append(edited_image["achieved_arousal"])

        if not valences or not arousals:
            print("No valid data found.")
            return

        valence_mean, valence_std = np.mean(valences), np.std(valences)
        arousal_mean, arousal_std = np.mean(arousals), np.std(arousals)

        return (valence_mean, valence_std), (arousal_mean, arousal_std)

    def get_lpips():

        loss_fn = lpips.LPIPS(net='vgg').to(DEVICE)

        transform = transforms.Compose([

            transforms.ToTensor(),
            # Normalize to [-1,1] as required by LPIPS
            transforms.Normalize([0.5], [0.5])
        ])

        original_images_loader = DataLoader(
            ImageDataset(dataset_path, transform=transform),
            batch_size=1, shuffle=False
        )

        results = []

        for emotion_id in range(len(valences)*len(arousals)):
            emotion_edited_images_loader = DataLoader(
                ImageDataset(output_path, transform=transform,
                             emotion_id=emotion_id),
                batch_size=1, shuffle=False
            )

            with torch.no_grad():
                for original_batch, edited_batch in zip(original_images_loader, emotion_edited_images_loader):
                    original_batch = original_batch.to(DEVICE)
                    edited_batch = edited_batch.to(DEVICE)

                    lpips_score = loss_fn(original_batch, edited_batch)
                    results.append(lpips_score.item())

        avg_lpips = np.mean(results)
        return avg_lpips
    # print(get_fid())
    # print(get_va())
    print(get_lpips())


if __name__ == '__main__':
    analyse_model_performance(
        "./results/result-2025-02-22-17-47",
        isGenerated=True
    )
