import matplotlib.pyplot as plt
from pathlib import Path

def plot_image_and_mask(image: Path | str, mask: Path | str):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title("Input image")
    ax[0].imshow(image)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask class ({i + 1})')
        ax[i + 1].imshow(mask == 1)
    plt.xticks([]), plt.yticks([])
    plt.show()