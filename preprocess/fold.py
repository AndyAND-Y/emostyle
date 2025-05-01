import os
import shutil

from tqdm import tqdm
from utils.imageDataset import ImageDatasetAFF
from concurrent.futures import ProcessPoolExecutor, as_completed


def renumber(folder_path):

    output_path = os.path.join(folder_path, "ordered")

    dataset = ImageDatasetAFF(folder_path=folder_path, load_every_n=1)

    os.makedirs(output_path, exist_ok=True)

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:

        futures = [
            executor.submit(save_image_task, *(image_path, output_path, index)) for index, image_path in enumerate(dataset.image_paths)
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Renumbering Images"):
            results = future.result()


def save_image_task(source_file_path, output_path, index):

    new_file_name = f"{index:06d}.jpg"
    output_file_path = os.path.join(output_path, new_file_name)
    shutil.copy2(source_file_path, output_file_path)


def main():

    folder_path = 'D:\\ML\\data\\ai-upscaled'
    renumber(folder_path)


if __name__ == '__main__':
    main()
