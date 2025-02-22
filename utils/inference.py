import multiprocessing
import os
import argparse
import random
import time
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from models.emo_mapping import EmoMappingWplus
from models.emonet import EmoNet
from models.stylegan2_interface import StyleGAN2
import json
import threading
from concurrent.futures import ThreadPoolExecutor


is_cuda = torch.cuda.is_available()
device = 'cuda' if is_cuda else 'cpu'
EMO_EMBED = 64
STG_EMBED = 512
INPUT_SIZE = 1024


def load_models(
        stylegan2_checkpoint_path="./pretrained/ffhq2.pkl",
        emostyle_checkpoint_path="./checkpoints/emo_mapping_wplus_2.pt",
        emonet8_checkpoint_path="./pretrained/emonet_8.pth"
):
    """
    Load the StyleGAN2, EmoNet, and EmoMappingWplus models.
    """
    # Load StyleGAN2
    stylegan = StyleGAN2(
        checkpoint_path=stylegan2_checkpoint_path,
        stylegan_size=INPUT_SIZE,
        is_dnn=True,
        is_pkl=True
    )
    stylegan.eval()
    stylegan.requires_grad_(False)

    # Load EmoNet
    ckpt_emo = torch.load(emonet8_checkpoint_path, weights_only=True)
    ckpt_emo = {k.replace('module.', ''): v for k, v in ckpt_emo.items()}
    emonet = EmoNet(n_expression=8)
    emonet.load_state_dict(ckpt_emo)
    emonet.eval()

    # Load EmoMappingWplus
    ckpt_emo_mapping = torch.load(
        emostyle_checkpoint_path,
        weights_only=True
    )
    emo_mapping = EmoMappingWplus(INPUT_SIZE, EMO_EMBED, STG_EMBED)
    emo_mapping.load_state_dict(ckpt_emo_mapping['emo_mapping_state_dict'])
    emo_mapping.eval()

    return {
        "stylegan2": stylegan,
        "emostyle": emo_mapping,
        "emonet": emonet
    }


def load_images_path(folder_path, limit):

    data = []

    for image_file in os.listdir(folder_path):

        if image_file.endswith(".png"):

            image_path = os.path.join(
                folder_path,
                image_file
            )
            latent_path = os.path.join(
                folder_path,
                os.path.splitext(image_file)[0] + '.npy'
            )

            data.append({
                "image_filename": image_file,
                "latent_path": latent_path,
                "image_path": image_path,
            })

            if (len(data) == limit):
                return data

    return data


def load_latent(latent_path):

    image_latent = np.load(latent_path, allow_pickle=False)
    image_latent = np.expand_dims(image_latent[:, :], 0)
    image_latent = torch.from_numpy(image_latent).float().cpu()

    return image_latent


def transform_tensor_image(image_tensor: torch.tensor):

    return np.clip(
        (image_tensor.detach().cpu().squeeze().numpy() * 255),
        0, 255
    ).transpose(1, 2, 0).astype(np.uint8)


def get_edited_image_data(latent, target_valence, target_arousal, emo_mapping, stylegan):

    emotion = torch.FloatTensor(
        [target_valence, target_arousal]
    ).unsqueeze(0).to(device)

    diff = emo_mapping(latent, emotion)

    fake_latents = latent + diff

    edited_image_tensor = stylegan.generate(fake_latents)
    edited_image_tensor = (edited_image_tensor + 1.0) / 2.0

    return edited_image_tensor


def compute_valence_arousal(image_tensor, model):

    emo_embed = model(image_tensor)

    valence = emo_embed[0][0].item()
    arousal = emo_embed[0][1].item()

    return valence, arousal


def save_image(image_data, path):
    image = Image.fromarray(image_data)
    image.save(path)


def save_images_concurrently(image_data_list, output_dir):

    with ThreadPoolExecutor() as executor:

        for image_data in image_data_list:
            filename = image_data['image_name']
            idx = image_data['idx']
            output_path = f"{output_dir}/{filename}/{filename}-{idx}.png"

            executor.submit(
                save_image, image_data['data'], output_path
            )


def save_result(result, path):
    with open(path, 'w') as json_file:
        json.dump(result, json_file, indent=4)


def save_results_concurrently(results, output_dir):

    threads = []

    for idx, result in enumerate(results):

        output_file_result = f"{
            output_dir}/{result['filename']}/result-{result['filename']}.json"

        thread = threading.Thread(
            target=save_result, args=(result, output_file_result)
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()


def process_single_image(image_id, idx, target_valence, target_arousal, latent, emostyle, stylegan2, emonet, out_folder_path, image_name):

    edited_image_tensor = get_edited_image_data(
        latent, target_valence, target_arousal, emostyle, stylegan2
    )

    achieved_valence, achieved_arousal = compute_valence_arousal(
        edited_image_tensor,
        emonet
    )

    output_file = f"{
        out_folder_path}/{image_name}/{image_name}-{idx}.png"

    data = {
        'data': transform_tensor_image(edited_image_tensor),
        "image_id": image_id,
        "idx": idx,
        "target_valence": target_valence,
        "target_arousal": target_arousal,
        "achieved_valence": achieved_valence,
        "achieved_arousal": achieved_arousal,
        "path": output_file,
        "image_name": image_name
    }

    return data


def consumer(results, queue: multiprocessing.Queue, semaphore, out_folder_path):

    futures = []

    with ThreadPoolExecutor() as executor:

        while True:

            semaphore.acquire()
            try:
                result = queue.get()

                if result == "STOP":

                    save_results_concurrently(results, out_folder_path)
                    break

                image_id = result['image_id']
                idx = result['idx']

                results[image_id]['edited_images'][idx] = {
                    "target_valence": result["target_valence"],
                    "target_arousal": result["target_arousal"],
                    "achieved_valence": result["achieved_valence"],
                    "achieved_arousal": result["achieved_arousal"],
                    "path": result["path"],
                }

                futures.append(executor.submit(
                    save_image, result['data'], result['path']))

            except Exception as e:
                print(f"Error in consumer: {e}")
                break

        # wait for all futures to finish
        for future in futures:
            future.result()


def producer(batches, output_folder, models, queue: multiprocessing.Queue, semaphore):

    for model in models:
        models[model] = models[model].to(device)

    batch_size = os.cpu_count()

    models['stylegan2'].generate(batches[0][4].to(device))

    t1 = time.time()

    for batch_start in tqdm(range(0, len(batches), batch_size), desc="Editing Images"):

        batch = batches[batch_start: batch_start + batch_size]

        for image_id, idx, target_valence, target_arousal, latent, image_name in batch:
            latent = latent.to(device)
            result = process_single_image(
                image_id,
                idx,
                target_valence,
                target_arousal,
                latent,
                models["emostyle"],
                models["stylegan2"],
                models["emonet"],
                output_folder,
                image_name
            )

            queue.put(result)
            semaphore.release()

    print(f"{time.time() - t1:.4f}s spent on task compute")


def run_edit_on_foder2(images_data, models, valences, arousals, out_folder_path, isRandom):

    os.makedirs(out_folder_path, exist_ok=True)
    emotions = [(v, a) for v in valences for a in arousals]

    # precompute latents
    latents = [
        load_latent(image_data['latent_path'])
        for image_data in images_data
    ]
    # precomupute outputs
    images_name = [
        f"{os.path.splitext(image_data["image_filename"])[0]}"
        for image_data in images_data
    ]

    for name in images_name:
        os.makedirs(f"{out_folder_path}/{name}/", exist_ok=True)

    # outputs
    results = [
        {
            "filename": images_name[idx],
            "original_image_path": image_data["image_path"],
            "edited_images": [0] * len(emotions)
        }
        for idx, image_data in enumerate(images_data)
    ]

    batches = [
        (
            image_id,
            id,
            valence if not isRandom else random.uniform(-1, 1),
            arousal if not isRandom else random.uniform(-1, 1),
            latents[image_id],
            images_name[image_id],
        )
        for image_id in range(len(images_data))
        for id, (valence, arousal) in enumerate(emotions)
    ]

    queue = multiprocessing.Queue()
    semaphore = multiprocessing.Semaphore(0)

    consumer_process = multiprocessing.Process(
        target=consumer, args=(results, queue, semaphore, out_folder_path)
    )
    producer_process = multiprocessing.Process(
        target=producer,
        args=(batches, out_folder_path, models, queue, semaphore)
    )

    consumer_process.start()
    producer_process.start()

    producer_process.join()
    queue.put("STOP")
    semaphore.release()

    consumer_process.join()


def generate_edited_images(images_path, stylegan2_checkpoint_path, checkpoint_path, output_path, valences, arousals, isRandom, limit: int = 100):
    """
    Main testing loop.
    """
    t1 = time.time()

    models = load_models(
        stylegan2_checkpoint_path,
        checkpoint_path
    )

    image_data = load_images_path(images_path, limit)

    results = run_edit_on_foder2(
        image_data,
        models,
        valences,
        arousals,
        output_path,
        isRandom
    )

    t2 = time.time()

    print(f"{t2-t1:.4f}s")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing script")
    parser.add_argument("--images_path", type=str, default="dataset/1024_pkl/")
    parser.add_argument("--stylegan2_checkpoint_path",
                        type=str, default="pretrained/ffhq2.pkl")
    parser.add_argument("--checkpoint_path", type=str,
                        default="checkpoints/emo_mapping_wplus_2.pt")
    parser.add_argument("--output_path", type=str, default="results")
    parser.add_argument("--valence", type=float, nargs='+',
                        default=[-1, -0.5, 0, 0.5, 1])
    parser.add_argument("--arousal", type=float, nargs='+',
                        default=[-1, -0.5, 0, 0.5, 1])
    parser.add_argument("--wplus", type=bool, default=True)
    parser.add_argument("--random", type=bool, default=True)

    args = parser.parse_args()

    generate_edited_images(
        images_path=args.images_path,
        stylegan2_checkpoint_path=args.stylegan2_checkpoint_path,
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        valences=args.valence,
        arousals=args.arousal,
    )
