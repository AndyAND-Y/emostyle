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

import lpips
import json
import os
import numpy as np
from torchvision import transforms
from torch.utils.data.dataloader import default_collate
from torchmetrics.image.fid import FrechetInceptionDistance
import utils.inference
import random
import time
import torch
from utils.imageDataset import ImageDataset
from torch.utils.data import  DataLoader
from models.arcface import ArcFace


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def uint8_transform(x):
    return (x * 255).clamp(0, 255).to(torch.uint8)


def collate_fn(batch):
    return default_collate(batch).to(DEVICE)


def analyse_model_performance(output_dir: str, test_mode="random", isGenerated=False, fromRandom=False):

    stylegan2_path = "pretrained/ffhq2.pkl"
    emostyle_path = "checkpoints/emo_mapping_wplus_2.pt"
    dataset_path = "dataset/1024_pkl/"
    arcface_path = "pretrained/model_ir_se50.pth"
    output_path = ""

    NUM_IMAGE = 1000
    valences = [0] if fromRandom else [-1, -0.5, 0, 0.5, 1]
    arousals = [0] if fromRandom else [-1, -0.5, 0, 0.5, 1]
    multiplier = 1

    if not isGenerated:

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
        ).to(DEVICE).eval()

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

        loss_fn = lpips.LPIPS(net='vgg', verbose=False).to(DEVICE)

        transform = transforms.Compose([

            transforms.ToTensor(),
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

    def get_id():

        arcface = ArcFace(arcface_path).to(DEVICE)
        arcface.eval()

        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

        original_images_loader = DataLoader(
            ImageDataset(dataset_path, transform=transform),
            shuffle=False
        )

        results = []

        for emotion_id in range(len(valences)*len(arousals)):
            emotion_edited_images_loader = DataLoader(
                ImageDataset(output_path, transform=transform,
                             emotion_id=emotion_id),
                shuffle=False
            )

            with torch.no_grad():
                for original_batch, edited_batch in zip(original_images_loader, emotion_edited_images_loader):
                    original_batch = original_batch.to(DEVICE)
                    edited_batch = edited_batch.to(DEVICE)

                    id_score = torch.nn.functional.cosine_similarity(
                        arcface(original_batch),
                        arcface(edited_batch)
                    )
                    results.append(id_score.item())

        return np.mean(results)

    results = {
        "ID": get_id(),
        "FID": get_fid(),
        "VA-std": get_va(),
        "LPIPS": get_lpips(),
    }
    for x in results:
        print(x, results[x])

    timestamp = time.strftime('%Y-%m-%d-%H-%M')
    results_filename = f"model-score-{timestamp}.json"
    results_filepath = os.path.join(output_dir, results_filename)

    with open(results_filepath, "w") as file:
        json.dump(results, file, indent=4)

    print(f"Results saved to {results_filepath}")

    for key, value in results.items():
        print(key, value)


if __name__ == '__main__':
    analyse_model_performance(
        "./results/result-2025-02-22-17-47",
        isGenerated=True,
        fromRandom=True,
    )
