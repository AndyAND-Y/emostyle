import os
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from models.emo_mapping import EmoMappingWplus
from models.emonet import EmoNet
from models.stylegan2_interface import StyleGAN2
import json
import threading

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
        map_location=torch.device(device),
        weights_only=True
    )
    emo_mapping = EmoMappingWplus(INPUT_SIZE, EMO_EMBED, STG_EMBED)
    emo_mapping.load_state_dict(ckpt_emo_mapping['emo_mapping_state_dict'])
    emo_mapping.to(device)
    emo_mapping.eval()

    # Move models to GPU if available
    if is_cuda:
        emo_mapping.cuda()
        stylegan.cuda()
        emonet.cuda()

    return {
        "stylegan2": stylegan,
        "emostyle": emo_mapping,
        "emonet": emonet
    }


def load_images_path(folder_path):

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

    return data


def load_latent(latent_path):

    image_latent = np.load(latent_path, allow_pickle=False)
    image_latent = np.expand_dims(image_latent[:, :], 0)
    image_latent = torch.from_numpy(image_latent).float().to(device)

    return image_latent


def transform_image_tenor(input_image: Image):

    input_image_tensor = torch.tensor(
        np.array(input_image).transpose(2, 0, 1) / 255.0,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    return input_image_tensor


def transform_tensor_image(image_tensor: torch.tensor):

    return np.clip(
        (image_tensor.detach().cpu().squeeze().numpy() * 255),
        0, 255
    ).astype(np.uint8).transpose(1, 2, 0)


def load_image(image_path: str):
    image = Image.open(image_path).convert('RGB')
    return image


def get_edited_image_data(latent, target_valence, target_arousal, emo_mapping, stylegan):
    """
    Generate an image with the given emotion (valence, arousal).
    """

    emotion = torch.FloatTensor(
        [target_valence, target_arousal]
    ).unsqueeze(0).to(device)

    fake_latents = latent + emo_mapping(latent, emotion)

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


def save_images_concurrently(image_data_list, output_dir, filename):
    threads = []

    for idx, image_data in enumerate(image_data_list):

        output_path = f"{
            output_dir}/{filename}/{filename}-{idx}.png"

        thread = threading.Thread(
            target=save_image, args=(image_data, output_path)
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()


def save_result(result, path):
    with open(path, 'w') as json_file:
        json.dump(result, json_file, indent=4)


def save_results_concurrently(results, output_dir):

    threads = []

    for idx, result in enumerate(results):

        output_file_result = f"{
            output_dir}/{result['filename']}/result-{result['filename']}.json"

        thread = threading.Thread(
            target=save_result, args=(results, output_file_result)
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()


def run_edit_on_foder(images_data, models, valences, arousals, out_folder_path):
    """
    Generate images by varying emotions (valence, arousal).
    """
    os.makedirs(out_folder_path, exist_ok=True)
    emotions = [(v, a) for v in valences for a in arousals]

    # precompute latents
    lantens = [
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

    for image_idx in tqdm(range(len(images_data)), desc="Editing Images"):

        latent = lantens[image_idx]

        image_data_list = [0] * len(emotions)

        for idx, (target_valence, target_arousal) in enumerate(emotions):

            edited_image_tensor = get_edited_image_data(
                latent, target_valence, target_arousal, models['emostyle'], models['stylegan2']
            )

            achieved_valence, achieved_arousal = compute_valence_arousal(
                edited_image_tensor,
                models['emonet']
            )

            img_name = images_name[image_idx]

            output_file = f"{
                out_folder_path}/{img_name}/{img_name}-{idx}.png"

            image_data_list[idx] = transform_tensor_image(edited_image_tensor)

            results[image_idx]["edited_images"][idx] = {
                "target_valence": target_valence,
                "target_arousal": target_arousal,
                "achieved_valence": achieved_valence,
                "achieved_arousal": achieved_arousal,
                "path": output_file
            }

        save_images_concurrently(
            image_data_list, out_folder_path, images_name[image_idx]
        )

    save_results_concurrently(results, out_folder_path)

    return results


def test(images_path, stylegan2_checkpoint_path, checkpoint_path, output_path, valences, arousals):
    """
    Main testing loop.
    """

    models = load_models(
        stylegan2_checkpoint_path,
        checkpoint_path
    )

    image_data = load_images_path(images_path)[:16]

    # return
    results = run_edit_on_foder(
        image_data,
        models,
        valences,
        arousals,
        output_path
    )

    print(len(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing script")
    parser.add_argument("--images_path", type=str, default="dataset/1024_pkl/")
    parser.add_argument("--stylegan2_checkpoint_path",
                        type=str, default="pretrained/ffhq2.pkl")
    parser.add_argument("--checkpoint_path", type=str,
                        default="checkpoints/emo_mapping_wplus_2.pt")
    parser.add_argument("--output_path", type=str, default="results-test/")
    parser.add_argument("--valence", type=float, nargs='+',
                        default=[-1, 0.5, 0, 0.5, 1])
    parser.add_argument("--arousal", type=float, nargs='+',
                        default=[-1, 0.5, 0, 0.5, 1])
    parser.add_argument("--wplus", type=bool, default=True)

    args = parser.parse_args()

    test(
        images_path=args.images_path,
        stylegan2_checkpoint_path=args.stylegan2_checkpoint_path,
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        valences=args.valence,
        arousals=args.arousal,
    )
