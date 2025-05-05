import json
import random
import pandas as pd
import torch
import matplotlib.pyplot as plt
import argparse


def plot_training_losses(loss_path: str, title: str):

    window_size = 100
    print(f"Loading losses from: {loss_path}")


    with open(loss_path, 'r') as f:
        data = json.load(f)

    all_losses = data.get('losses')

    losses = ["id_loss", "emo_loss", "landmarks", "latent_loss", "recon_loss", "latent_reg", "sum_non_gan", "bg_loss"]

    loss_values = {loss_name: [] for loss_name in losses}

    for i, loss_log in enumerate(all_losses):

        for loss_name, loss_str_value in loss_log.items():
        
            loss_float_value = float(loss_str_value)
            loss_values[loss_name].append(loss_float_value)

    print(f"Found loss keys: {list(loss_values.keys())}")
    


    # Plot each loss key separately
    for loss_name, values in loss_values.items():

        loss_series = pd.Series(values)

        print(len(loss_series))

        smoothed_values = loss_series.rolling(
            window=window_size, min_periods=window_size
        ).mean()

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(smoothed_values)), smoothed_values, label=loss_name)
        plt.xlabel("Iteration")
        plt.ylabel(loss_name)
        plt.title(f"{title} Training Loss: {loss_name}")
        plt.ylim(smoothed_values.min() * 0.9, smoothed_values.max() * 1.1)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        save_path = f"C:\\Users\\Andy\\Desktop\\UNI\\res\\training_data_charts\\{title}_{loss_name}.png"
        plt.savefig(save_path)
    plt.show()  # Display all generated plots



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot training losses from saved checkpoints")

    parser.add_argument("--path", required=True,
                        help="Path to the JSON file containing training losses")

    parser.add_argument("--title", required=True,
                        help="Title")

    args = parser.parse_args()

    plot_training_losses(
        loss_path=args.path,
        title=args.title
    )
