import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from models.emo_mapping import EmoMappingWplus
from models.emonet import EmoNet
from models.stylegan2_interface import StyleGAN2


is_cuda = torch.cuda.is_available()
device = 'cuda' if is_cuda else 'cpu'
EMO_EMBED = 64
STG_EMBED = 512
INPUT_SIZE = 1024


def load_models(
        stylegan2_checkpoint_path="pretrained/ffhq2.pkl",
        emostyle_checkpoint_path="checkpoints/emo_mapping_wplus_2.pt",
        emonet8_checkpoint_path="pretrained/emonet_8.pth"
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

    return stylegan, emo_mapping, emonet


def load_latents(images_path, test_mode, wplus):
    """
    Load image latents from files based on the test mode.
    """
    latents = {}

    if test_mode == 'folder_images':

        for image_file in os.listdir(images_path):

            if image_file.endswith(".png"):

                image_path = os.path.join(
                    images_path,
                    image_file
                )
                latent_path = os.path.join(
                    images_path,
                    os.path.splitext(image_file)[0] + '.npy'
                )

                image_latent = np.load(latent_path, allow_pickle=False)

                if wplus:
                    image_latent = np.expand_dims(image_latent[:, :], 0)
                else:
                    image_latent = np.expand_dims(image_latent[0, :], 0)

                image_latent = torch.from_numpy(image_latent).float()
                latents[image_file] = {
                    "lantent": image_latent,
                    "image_path": image_path,
                    "latent_path": latent_path
                }

        return latents

    return latents


def generate_images(latents, emo_mapping, stylegan, valence, arousal, emonet, output_path):
    """
    Generate images by varying emotions (valence, arousal).
    """
    emos_data = {}
    num_images = len(valence) * len(arousal)

    for v in range(len(valence)):
        for a in range(len(arousal)):
            emos_data[(valence[v], arousal[a])] = []

    for img, data in latents.items():

        latent = data["lantent"]
        image_path = data["image_path"]
        image_name = img

        input_image = Image.open(image_path).convert('RGB')

        inpute_image_tensor = torch.tensor(
            np.array(input_image).transpose(2, 0, 1) / 255.0,
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        original_emo_embed = emonet(inpute_image_tensor)

        _, ax_g = plt.subplots(1, num_images + 1, figsize=(100, 50))
        plt.subplots_adjust(left=.05, right=.95, wspace=0, hspace=0)
        iter = 1

        ax_g[0].imshow(input_image)
        ax_g[0].set_title("Original Image", fontsize=80)
        ax_g[0].axis('off')

        original_valence = original_emo_embed[0][0].item()
        original_arousal = original_emo_embed[0][1].item()
        print(original_valence, original_valence)

        text = f"Original V:{original_valence:.2f} A:{original_arousal:.2f}"

        ax_g[0].text(
            1, 0,
            text,
            fontsize=40, color='white',
            ha='right', va='bottom',
            transform=ax_g[0].transAxes,
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none')
        )

        image_tensors = []

        for v_idx in tqdm(range(len(valence)), desc="Valences"):
            for a_idx in range(len(arousal)):

                emotion = torch.FloatTensor(
                    [valence[v_idx], arousal[a_idx]]).unsqueeze_(0).to(device)

                latent = latent.to(device)

                fake_latents = latent + emo_mapping(latent, emotion)

                generated_image_tensor = stylegan.generate(fake_latents)
                generated_image_tensor = (generated_image_tensor + 1.) / 2.

                emo_embed = emonet(generated_image_tensor)

                achieved_valence = emo_embed[0][0].item()
                achieved_arousal = emo_embed[0][1].item()

                generated_image = generated_image_tensor.detach().cpu().squeeze().numpy()

                generated_image = np.clip(
                    generated_image * 255, 0, 255
                ).astype(np.int32)

                generated_image = generated_image.transpose(
                    1, 2, 0
                ).astype(np.uint8)

                emos_data[(valence[v_idx], arousal[a_idx])].append(
                    {
                        "target_valence": valence[v_idx],
                        "target_arousal": arousal[a_idx],
                        "achieved_valence": achieved_valence,
                        "achieved_arousal": achieved_arousal,
                    }
                )

                image_tensors.append(generated_image_tensor)

                ax_g[iter].imshow(generated_image)
                ax_g[iter].set_title(
                    f"V: {achieved_valence:.2f}, A: {achieved_arousal:.2f}",
                    fontsize=60
                )
                ax_g[iter].axis('off')

                text = f"Target V:{valence[v_idx]:.2f} A:{arousal[a_idx]:.2f}"

                ax_g[iter].text(
                    1, 0,
                    text,
                    fontsize=40, color='white',
                    ha='right', va='bottom',
                    transform=ax_g[iter].transAxes,
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none')
                )

                iter += 1

        result_image_path = os.path.join(output_path, f"result-{image_name}")
        plt.savefig(result_image_path, bbox_inches='tight')
        plt.close()

    return emos_data


def test(images_path, stylegan2_checkpoint_path, checkpoint_path, output_path, test_mode, valence, arousal, wplus):
    """
    Main testing loop.
    """
    stylegan, emo_mapping, emonet = load_models(
        stylegan2_checkpoint_path,
        checkpoint_path
    )

    latents = load_latents(images_path, test_mode, wplus)

    emos_data = generate_images(
        latents,
        emo_mapping,
        stylegan,
        valence,
        arousal,
        emonet,
        output_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing script")
    parser.add_argument("--images_path", type=str, default="dataset/1024_pkl/")
    parser.add_argument("--stylegan2_checkpoint_path",
                        type=str, default="pretrained/ffhq2.pkl")
    parser.add_argument("--checkpoint_path", type=str,
                        default="checkpoints/emo_mapping_wplus_2.pt")
    parser.add_argument("--output_path", type=str, default="results/")
    parser.add_argument("--test_mode", type=str, default="folder_images")
    parser.add_argument("--valence", type=float, nargs='+', default=[-1, 0, 1])
    parser.add_argument("--arousal", type=float, nargs='+', default=[-1, 0, 1])
    parser.add_argument("--wplus", type=bool, default=True)

    args = parser.parse_args()

    test(
        images_path=args.images_path,
        stylegan2_checkpoint_path=args.stylegan2_checkpoint_path,
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        test_mode=args.test_mode,
        valence=args.valence,
        arousal=args.arousal,
        wplus=args.wplus
    )
