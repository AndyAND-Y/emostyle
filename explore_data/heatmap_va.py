import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_heatmap(data):
    df = pd.DataFrame(data, columns=['x', 'y'])
    print(df.describe())

    num_bins = 100

    hist, xedges, yedges = np.histogram2d(
        np.array(data)[:, 0],
        np.array(data)[:, 1],
        bins=num_bins,
        range=[[-1, 1], [-1, 1]],
    )

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    sns.histplot(df, x='x', y='y', bins=num_bins, cmap='coolwarm', cbar=True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Heatmap of Valence Arousal Pairs - Upscaled')
    plt.show()


def read_va_results(file_path):
    va_data = []

    missread_values = 0

    with open(file_path, 'r') as f:
        for line in f:
            match = re.search(
                r": ([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?), ([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)", line)
            if match:
                valence_str, arousal_str = match.groups()
                valence = float(valence_str)
                arousal = float(arousal_str)

                if valence > 1 or valence < -1 or arousal > 1 or arousal < -1:
                    missread_values += 1

                    continue
                va_data.append((valence, arousal))

    return va_data


if __name__ == '__main__':

    results_path = "D:\\ML\\data\\upscaled\\ordered-1024x1024\\training\\va_results.txt"
    va_data = read_va_results(results_path)
    create_heatmap(va_data)
