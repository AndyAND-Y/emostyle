import os
import shutil
import argparse


def clear_folder(folder_path):
    if os.path.exists(folder_path):
        print(f"Clearing destination folder: '{folder_path}'...")
        try:
            shutil.rmtree(folder_path)
            print("Folder cleared.")
        except OSError as e:
            print(f"Error clearing folder {folder_path}: {e}")
            return False
    os.makedirs(folder_path, exist_ok=True)
    return True


def copy_every_nth_pair(source_folder, destination_folder, n):

    if not os.path.exists(source_folder):
        print(f"Error: Source folder '{source_folder}' not found.")
        return

    if not clear_folder(destination_folder):
        print("Aborting due to error clearing destination folder.")
        return

    files = os.listdir(source_folder)

    base_names = set()
    for file in files:
        if '.' in file:
            base_name, ext = os.path.splitext(file)
            base_names.add(base_name)

    sorted_base_names = sorted(list(base_names))

    print(f"Found {len(sorted_base_names)} unique base names in source.")
    print(f"Copying every {n}-th pair to '{destination_folder}'...")

    copied_count = 0
    for i, base_name in enumerate(sorted_base_names):
        if i % n == 0:
            jpg_file = f"{copied_count:06}.jpg"
            npy_file = f"{copied_count:06}.npy"

            src_jpg_path = os.path.join(source_folder, jpg_file)
            src_npy_path = os.path.join(source_folder, npy_file)

            dest_jpg_path = os.path.join(destination_folder, jpg_file)
            dest_npy_path = os.path.join(destination_folder, npy_file)

            shutil.copy2(src_jpg_path, dest_jpg_path)
            shutil.copy2(src_npy_path, dest_npy_path)

            copied_count += 1

    print(
        f"\nFinished copying. Copied {copied_count} pairs to '{destination_folder}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clear destination and copy every n-th image and latent pair from a folder.")
    parser.add_argument(
        "source_folder", help="The path to the source folder containing image and latent files.")
    parser.add_argument(
        "destination_folder", help="The path to the destination folder where files will be copied (will be cleared first).")
    parser.add_argument(
        "n", type=int, help="Copy every n-th pair (e.g., 3 for every 3rd pair).")

    args = parser.parse_args()

    if args.n <= 0:
        print("Error: 'n' must be a positive integer.")
    else:
        copy_every_nth_pair(args.source_folder,
                            args.destination_folder, args.n)
