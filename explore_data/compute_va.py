import os
import queue
import threading
from models.emonet import EmoNet
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.imageDataset import ImageDatasetAFF

emonet8_checkpoint_path = "./pretrained/emonet_8.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_emonet():
    # Load EmoNet
    ckpt_emo = torch.load(emonet8_checkpoint_path, weights_only=True)
    ckpt_emo = {k.replace('module.', ''): v for k, v in ckpt_emo.items()}
    emonet = EmoNet(n_expression=8)
    emonet.load_state_dict(ckpt_emo)

    emonet = torch.compile(emonet, backend="cudagraphs")

    emonet.eval()
    emonet = emonet.to(device)

    return emonet


def compute_va_batch(image_tensors, emonet):
    emo_embed = emonet(image_tensors)
    valence = emo_embed[:, 0].cpu().numpy().tolist()
    arousal = emo_embed[:, 1].cpu().numpy().tolist()
    return valence, arousal


def write_results_to_file(results_queue, output_file):
    with open(output_file, "w") as f:
        while True:
            result = results_queue.get()
            results_queue.task_done()
            if result is None:  # Sentinel value to stop the thread
                break
            path, valence, arousal = result
            f.write(f"{path}: {valence}, {arousal}\n")


def compute_va_folder(folder_path, output_file):
    emonet = load_emonet()
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = ImageDatasetAFF(
        folder_path, transform=preprocess, load_every_n=1
    )

    dataloader = DataLoader(
        dataset,
        batch_size=40,
        shuffle=False,
        num_workers=os.cpu_count()//4,
        pin_memory=True
    )

    results_queue = queue.Queue()
    writer_thread = threading.Thread(
        target=write_results_to_file, args=(results_queue, output_file)
    )
    writer_thread.daemon = True  # Allow main thread to exit even if writer is blocked
    writer_thread.start()

    for batch in tqdm(dataloader, desc="Processing batches"):
        images, paths = batch
        images = images.to(device)
        with torch.no_grad():
            valence, arousal = compute_va_batch(images, emonet)

        for i, path in enumerate(paths):
            results_queue.put((path, valence[i], arousal[i]))

    # Signal the writer thread to stop
    results_queue.put(None)
    writer_thread.join()  # Wait for the writer thread to finish


def main():

    folder_path = "D:\\ML\\data\\upscaled\\ordered-1024x1024\\training"
    output_file = os.path.join(folder_path, "va_results.txt")
    compute_va_folder(folder_path, output_file)
    print(f"Valence and arousal results written to: {output_file}")


if __name__ == "__main__":
    main()
