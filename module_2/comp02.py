"""
Class containing code related to questions for assignment 2
"""
import copy
import numpy as np
import os
from matplotlib import pyplot as plt
import skimage.io as skio
import skimage.filters as skfl
import skimage.color as skcol

class Histogram:
    def __init__(self):
        pass

    def contrast_stretch(self, img, lower_limit=0, upper_limit=1, lowest_pixel=None, highest_pixel=None, flatten=False):
        """Performs contrast stretching

        Args:
            img (np.array): Image array (grayscale).
            lower_limit (int, optional): Lower limit of stretching. Defaults to 0.
            upper_limit (int, optional): Upper limit of stretching. Defaults to 1.
            lowest_pixel (int, optional): Value of lowest pixel in the image. If None, it is calculated. Defaults to None.
            highest_pixel (int, optional): Value of highest pixel in the image. If None, it is calculated. Defaults to None.
            flatten (bool, optional): If True, flattens the image array. Defaults to False.
        """
        img_new = copy.deepcopy(img)
        if lowest_pixel is None:
            x_min, y_min = np.unravel_index(np.argmin(img, axis=None), img.shape)
            lowest_pixel = img[x_min, y_min]
        if highest_pixel is None:
            x_max, y_max = np.unravel_index(np.argmax(img, axis=None), img.shape)
            highest_pixel = img[x_max, y_max]

        scaling_factor = (upper_limit - lower_limit)/(highest_pixel - lowest_pixel)
        for row_num, _ in enumerate(img):
            for col_num, _ in enumerate(img[row_num]):
                img_new[row_num, col_num] = (img[row_num, col_num] - lowest_pixel) * scaling_factor + lower_limit

        if flatten:
            return img_new.flatten()

        return img_new

    def pipeline(self, filepath_img):
        """Performs mask detection and histogram stretching.

        Args:
            filepath_img (str): filepath of the image
        """
        fig, axes = plt.subplots(2, 2)
        fig.set_figheight(10)
        fig.set_figwidth(15)

        im = skcol.rgb2gray(skio.imread(filepath_img))
        threshold_otsu = skfl.threshold_otsu(im)

        fig.suptitle(f"Contrast stretching of image: {os.path.basename(filepath_img)} | Otsu threshold: {str(round(threshold_otsu, 2))}")

        list_img = [im, self.contrast_stretch(im, lowest_pixel=threshold_otsu)]
        list_title = ["Original", "After contrast stretching"]

        for i in [0, 1]:
            ax = axes[i, 0].imshow(list_img[i], cmap="gray", vmin=0, vmax=1)
            plt.colorbar(ax, ax=axes[i, 0])
            axes[i, 0].set_title(list_title[i] + ": Image")
            axes[i, 1].hist(list_img[i].flatten(), bins=20, range=(0, 1))
            axes[i, 1].set_title(list_title[i] + ": Histogram")

        return fig, threshold_otsu