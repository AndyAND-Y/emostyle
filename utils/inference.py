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


def compute_image_tenor(input_image: Image):

    input_image_tensor = torch.tensor(
        np.array(input_image).transpose(2, 0, 1) / 255.0,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    return input_image_tensor


def compute_tensor_image(image_tensor: torch.tensor):

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

    edited_image_data = np.clip(
        (edited_image_tensor.detach().cpu().squeeze().numpy() * 255),
        0, 255
    ).astype(np.uint8).transpose(1, 2, 0)

    return edited_image_data


def compute_valence_arousal(image_tensor, model):

    emo_embed = model(image_tensor)

    valence = emo_embed[0][0].item()
    arousal = emo_embed[0][1].item()

    return valence, arousal


def save_image(image_data, path):
    image = Image.fromarray(image_data)
    image.save(path)


def run_edit_on_foder(images_data, models, valences, arousals, out_folder_path):
    """
    Generate images by varying emotions (valence, arousal).
    """

    if not os.path.exists(out_folder_path):
        os.makedirs(out_folder_path, exist_ok=True)

    results = []

    for image_idx in tqdm(range(len(images_data)), desc="Editing Images"):

        latent = load_latent(images_data[image_idx]['latent_path'])

        image_results = {
            "original_image_path": images_data[image_idx]["image_path"],
            "edited_images": []
        }

        img_name = os.path.splitext(
            images_data[image_idx]["image_filename"]
        )[0]

        output_dir = f"{out_folder_path}/{img_name}/"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        save_idx = 0

        for target_valence in valences:
            for target_arousal in arousals:

                emotion = torch.FloatTensor(
                    [target_valence, target_arousal]
                ).unsqueeze_(0).to(device)

                fake_latents = latent + models["emostyle"](latent, emotion)

                edited_image_tensor = models['stylegan2'].generate(
                    fake_latents)
                edited_image_tensor = (edited_image_tensor + 1.) / 2.

                achieved_valence, achieved_arousal = compute_valence_arousal(
                    edited_image_tensor,
                    models["emonet"]
                )

                edited_image_data = edited_image_tensor.detach().cpu().squeeze().numpy()

                edited_image_data = np \
                    .clip(
                        edited_image_data * 255, 0, 255
                    )\
                    .transpose(
                        1, 2, 0
                    )\
                    .astype(np.uint8)

                output_file = f"{
                    out_folder_path}/{img_name}/{img_name}-{save_idx}.png"

                save_image(
                    edited_image_data,
                    output_file
                )

                image_results["edited_images"].append({
                    "target_valence": target_valence,
                    "target_arousal": target_arousal,
                    "achieved_valence": achieved_valence,
                    "achieved_arousal": achieved_arousal,
                    "path": output_file
                })

                save_idx += 1

        results.append(image_results)

        output_file_result = f"{
            out_folder_path}/{img_name}/result-{img_name}.json"

        with open(output_file_result, 'w') as json_file:
            json.dump(image_results, json_file, indent=4)

    return results


def test(images_path, stylegan2_checkpoint_path, checkpoint_path, output_path, valences, arousals):
    """
    Main testing loop.
    """

    models = load_models(
        stylegan2_checkpoint_path,
        checkpoint_path
    )

    image_data = load_images_path(images_path)[:5]

    # return
    results = run_edit_on_foder(
        image_data,
        models,
        valences,
        arousals,
        output_path
    )

    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing script")
    parser.add_argument("--images_path", type=str, default="dataset/1024_pkl/")
    parser.add_argument("--stylegan2_checkpoint_path",
                        type=str, default="pretrained/ffhq2.pkl")
    parser.add_argument("--checkpoint_path", type=str,
                        default="checkpoints/emo_mapping_wplus_2.pt")
    parser.add_argument("--output_path", type=str, default="results/")
    parser.add_argument("--valence", type=float, nargs='+', default=[-1, 0, 1])
    parser.add_argument("--arousal", type=float, nargs='+', default=[-1, 0, 1])
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
