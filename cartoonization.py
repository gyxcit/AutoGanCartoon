import cv2
import scipy
from scipy import stats
import numpy as np
from collections import defaultdict
import os
from pathlib import Path

def update_c(C, hist):
    """
    Updates the centers C by grouping indices from the histogram.

    :param C: Array of current centers.
    :param hist: Histogram to analyze.
    :return: Updated centers and associated groups.
    """
    while True:
        groups = defaultdict(list)

        # Assign histogram indices to the nearest center
        for i in range(len(hist)):
            if hist[i] == 0:
                continue
            d = np.abs(C - i)
            index = np.argmin(d)
            groups[index].append(i)

        # Calculate new centers
        new_C = np.array(C)
        for i, indices in groups.items():
            if np.sum(hist[indices]) == 0:
                continue
            new_C[i] = int(np.sum(indices * hist[indices]) / np.sum(hist[indices]))

        # Stop if centers no longer change
        if np.sum(new_C - C) == 0:
            break
        C = new_C

    return C, groups

def K_histogram(hist):
    """
    Partitions a histogram into clusters using K-modes.

    :param hist: Histogram to partition.
    :return: Final cluster centers.
    """
    alpha = 0.001  # Threshold for normality test
    N = 80  # Minimum size for a cluster
    C = np.array([128])  # Initial center

    while True:
        C, groups = update_c(C, hist)

        new_C = set()
        for i, indices in groups.items():
            if len(indices) < N:
                new_C.add(C[i])
                continue

            # Perform a normality test on the data in the cluster
            z, pval = stats.normaltest(hist[indices])
            if pval < alpha:
                left = 0 if i == 0 else C[i - 1]
                right = len(hist) - 1 if i == len(C) - 1 else C[i + 1]
                delta = right - left
                if delta >= 3:
                    c1 = (C[i] + left) / 2
                    c2 = (C[i] + right) / 2
                    new_C.add(c1)
                    new_C.add(c2)
                else:
                    new_C.add(C[i])
            else:
                new_C.add(C[i])
        if len(new_C) == len(C):
            break
        else:
            C = np.array(sorted(new_C))
    return C

def caart(img, target_size=None):
    """
    Applies a cartoon effect to an image.

    :param img: Input image.
    :param target_size: Target size to resize the image.
    :return: Image with a cartoon effect.
    """
    if target_size:
        img = cv2.resize(img, target_size)

    kernel = np.ones((2, 2), np.uint8)  # Kernel for erosion
    output = np.array(img)
    x, y, c = output.shape

    # Apply bilateral filter
    for i in range(c):
        output[:, :, i] = cv2.bilateralFilter(output[:, :, i], 5, 150, 150)

    # Detect edges
    edge = cv2.Canny(output, 100, 200)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2HSV)

    hists = []

    # Compute histograms for each channel
    hist, _ = np.histogram(output[:, :, 0], bins=np.arange(180 + 1))
    hists.append(hist)
    hist, _ = np.histogram(output[:, :, 1], bins=np.arange(256 + 1))
    hists.append(hist)
    hist, _ = np.histogram(output[:, :, 2], bins=np.arange(256 + 1))
    hists.append(hist)

    # Cluster the histogram values
    C = []
    for h in hists:
        C.append(K_histogram(h))

    # Map pixel values to the closest cluster center
    output = output.reshape((-1, c))
    for i in range(c):
        channel = output[:, i]
        index = np.argmin(np.abs(channel[:, np.newaxis] - C[i]), axis=1)
        output[:, i] = C[i][index]
    output = output.reshape((x, y, c))
    output = cv2.cvtColor(output, cv2.COLOR_HSV2RGB)

    # Draw contours
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(output, contours, -1, 0, thickness=1)

    # Apply erosion to smoothen
    for i in range(3):
        output[:, :, i] = cv2.erode(output[:, :, i], kernel, iterations=1)

    return output

def cartoonize_folder(input_folder, output_folder, target_size=None):
    """
    Applies the cartoon effect to all images in a folder.

    :param input_folder: Path to the folder containing input images.
    :param output_folder: Path to save the cartoonized images.
    :param target_size: Target size to resize the images (optional).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # Skip non-image files
        if not (filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg')):
            continue

        try:
            img = cv2.imread(file_path)
            cartoonized_img = caart(img, target_size)

            output_filename = f"{os.path.splitext(filename)[0]}{os.path.splitext(filename)[1]}"
            output_path = os.path.join(output_folder, output_filename)

            cv2.imwrite(output_path, cartoonized_img)
            print(f"Successfully cartoonized and saved: {output_path}")
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

# Example usage
input_folder = Path(r"dataset/normal")
output_folder = Path(r"dataset/cartoon")
target_size = (256, 256)  # Resize images to 256x256 pixels

cartoonize_folder(input_folder, output_folder, target_size=target_size)
