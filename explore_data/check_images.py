
import json
import math
import random
from PIL import Image
import lpips
from matplotlib import pyplot as plt
import torch
from torchvision import transforms
from models.vggface2 import VGGFace2

PATH_INPUT = "C:\\Users\\Andy\\Desktop\\UNI\\project\\emostyle\\dataset\\1024_pkl"
PATH_REF = "C:\\Users\\Andy\\Desktop\\UNI\\project\\emostyle\\results\\result-2025-05-01-18-26"
PATH_BICUBIC = "C:\\Users\\Andy\\Desktop\\UNI\\project\\emostyle\\results\\result-2025-05-01-18-55"
PATH_AI = "C:\\Users\\Andy\\Desktop\\UNI\\project\\emostyle\\results\\result-2025-05-01-18-40"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


lpips_model = lpips.LPIPS(net='vgg', verbose=False).to(DEVICE)
vggface2_path = 'pretrained/resnet50_ft_weight.pkl' 
id_model = VGGFace2(vggface2_path).to(DEVICE).eval()

def get_images(id):

    image_input = Image.open(f"{PATH_INPUT}\\{id}.png")
    image_ref = Image.open(f"{PATH_REF}\\{id}\\{id}-0.png")
    image_bicubic = Image.open(f"{PATH_BICUBIC}\\{id}\\{id}-0.png")
    image_ai = Image.open(f"{PATH_AI}\\{id}\\{id}-0.png")

    return {
        "input": image_input,
        "ref": image_ref,
        "bicubic": image_bicubic,
        "ai": image_ai
    }

def get_va_metadata(id):

    metadata_ref = json.load(open(f"{PATH_REF}\\{id}\\result-{id}.json"))["edited_images"][0]
    metadata_bicubic = json.load(open(f"{PATH_BICUBIC}\\{id}\\result-{id}.json"))["edited_images"][0]
    metadata_ai = json.load(open(f"{PATH_AI}\\{id}\\result-{id}.json"))["edited_images"][0]

    return {
        "ref": metadata_ref,
        "bicubic": metadata_bicubic,
        "ai": metadata_ai
    }


def additional_metrics(images):

    metrics = {}
    input_img_pil = images["input"]

   
    lpips_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
    ])
    
    id_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    input_img_lpips = lpips_transform(input_img_pil).unsqueeze(0).to(DEVICE)
    input_img_id = id_transform(input_img_pil).unsqueeze(0).to(DEVICE)

    input_id_embedding = id_model(input_img_id)

    # Process input image
    input_img_lpips = lpips_transform(input_img_pil).unsqueeze(0).to(DEVICE)
    input_img_id = id_transform(input_img_pil).unsqueeze(0).to(DEVICE)
    
    for key in ["ref", "bicubic", "ai"]:
        if key not in images or images[key] is None:
            metrics[key] = {"lpips": "N/A", "id": "N/A"}
            continue

        gen_img_pil = images[key]
        gen_img_lpips = lpips_transform(gen_img_pil).unsqueeze(0).to(DEVICE)
        gen_img_id = id_transform(gen_img_pil).unsqueeze(0).to(DEVICE)

        lpips_score = lpips_model(input_img_lpips, gen_img_lpips).item()

        gen_id_embedding = id_model(gen_img_id)
        id_score = torch.nn.functional.cosine_similarity(input_id_embedding, gen_id_embedding).item()

        metrics[key] = {"lpips": lpips_score, "id": id_score}
    
    return metrics

def vizualize_images(id):

    images = get_images(id)
    metadata = get_va_metadata(id)
    
    metrics = additional_metrics(images)

    fig, axes = plt.subplots(1, 4, figsize=(20, 8))

    titles = {
        "input": f"Input Image (ID: {id})",
        "ref": "Reference Model Output",
        "bicubic": "Bicubic Model Output",
        "ai": "AI Model Output"
    }

    for i, (key, img) in enumerate(images.items()):
        ax = axes[i]
        ax.imshow(img)

        title = titles[key]
        if(key in metadata):
            meta = metadata[key]
            va_info = (
                    f"\nTarget V: {meta.get('target_valence', 'N/A'):.2f}, "
                    f"A: {meta.get('target_arousal', 'N/A'):.2f}"
                    f"\nAchieved V: {meta.get('achieved_valence', 'N/A'):.2f}, "
                    f"A: {meta.get('achieved_arousal', 'N/A'):.2f}"
                    f"\nDist: {calculate_va_distance(
                        meta.get('target_valence', 0), meta.get('target_arousal', 0),
                        meta.get('achieved_valence', 0), meta.get('achieved_arousal', 0)
                    ):.4f}"
                )
            title += va_info

        if(key in metrics):
            metric_info = metrics[key]
            lpips_val = f"{metric_info.get('lpips', 'N/A'):.4f}" 
            id_val = f"{metric_info.get('id', 'N/A'):.4f}" 

            title += f"\nLPIPS: {lpips_val}, ID: {id_val}"

        ax.set_title(title, fontsize=8) 
        ax.axis('off') 

    plt.tight_layout() 
    # plt.show()

    plt.savefig(f"output\\{id}.png")

def calculate_va_distance(target_v, target_a, achieved_v, achieved_a):
    return math.sqrt((achieved_v - target_v)**2 + (achieved_a - target_a)**2)


seen = []

def get_good_image_id():

    while True:
        
        id = f"{random.randint(0, 999):06}"

        if id in seen:
            continue

        # metadata = get_va_metadata(id)
        
        # target_v = metadata["ref"].get("target_valence")
        # target_a = metadata["ref"].get("target_arousal")

        # achieved_v_ref = metadata["ref"].get("achieved_valence")
        # achieved_a_ref = metadata["ref"].get("achieved_arousal")

        # achieved_v_bicubic = metadata["bicubic"].get("achieved_valence")
        # achieved_a_bicubic = metadata["bicubic"].get("achieved_arousal")

        # achieved_v_ai = metadata["ai"].get("achieved_valence")
        # achieved_a_ai = metadata["ai"].get("achieved_arousal")

        # # Calculate distances
        # dist_ref = calculate_va_distance(target_v, target_a, achieved_v_ref, achieved_a_ref)
        # dist_bicubic = calculate_va_distance(target_v, target_a, achieved_v_bicubic, achieved_a_bicubic)
        # dist_ai = calculate_va_distance(target_v, target_a, achieved_v_ai, achieved_a_ai)

        # epsilon = 1e-3
        # if dist_ai < dist_ref - epsilon and dist_ai < dist_bicubic - epsilon:
        # print(f"Found good ID: {id} (AI distance: {dist_ai:.4f}, Ref distance: {dist_ref:.4f}, Bicubic distance: {dist_bicubic:.4f})")
        seen.append(id)
        return id 


def main():

    ids = ["000176", "000230", "000422", "000590", "000969", "000659", "000112", "000473", "000669", "000999", "000001", "000002", "000003", "000004", "000005", "000006", "000007", "000008", "000009", "000010"]
    for id in ids:
        vizualize_images(f"{id}")


if __name__ == '__main__':
    main()
