"""
Class for code related to Task 1
"""
import copy, os, yaml
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as skio
import skimage.draw as skdr
import skimage.util as skut
import skimage.filters as skfl
import skimage.transform as sktr
import skimage.color as skcol
import sklearn.model_selection as le_ms
import sklearn.preprocessing as le_pr
import sklearn.linear_model as le_lm
import sklearn.metrics as le_me

class Task1:
    def __init__(self):
        self.folderpath_root = "/Users/cmokashi/Documents/UTHealth/bmi_6331/data_challenge/"
        self.folderpath_results = os.path.join(self.folderpath_root, "dataset", "results")

    def get_pipelines(self):
        """Returns pipeline UIDs.
        """
        # Extract log files
        list_uids = [os.path.splitext(filename)[0] for filename in os.listdir(self.folderpath_root) if filename.endswith(".log")]
        
        list_uids_complete = []
        for filename in os.listdir(self.folderpath_results):
            if filename == ".DS_Store":
                continue
            filename = os.path.splitext(filename)[0]
            if "_" in filename:
                filename = filename.split("_")[0]
                list_uids_complete.append(filename)
        list_uids_complete = list(set(list_uids_complete))
        
        return list_uids, list_uids_complete

    def load_plot_roc(self):
        fig = plt.figure(figsize=(10, 10))
        fig.suptitle("ROC curves")
        list_uids, list_uids_complete = self.get_pipelines()
        for uid in list_uids_complete:
            fpr = np.load(os.path.join(self.folderpath_results, f"{uid}_fpr.npy")), 
            tpr = np.load(os.path.join(self.folderpath_results, f"{uid}_tpr.npy")), 
            thresholds = np.load(os.path.join(self.folderpath_results, f"{uid}_thresholds.npy"))
            with open(os.path.join(self.folderpath_results, f"{uid}.yaml"), "r") as f:
                dict_info = yaml.full_load(f)
            list_title = [f"AUC: {round(le_me.auc(fpr[0], tpr[0]), 3)}"]
            if dict_info["transfer_learning"] is not None:
                list_title.append(dict_info["transfer_learning"])
            list_title.append(dict_info["model"]["type"])
            str_params = "("
            for key in sorted(dict_info["model"]):
                if key != "type":
                    str_params += f"{key}: {dict_info['model'][key]} "
            str_params += ")"
            list_title.append(str_params)
            title = " | ".join(list_title)
            plt.plot(fpr[0], tpr[0], "-", label=title)
        plt.xlabel("1 - Specificity")
        plt.ylabel("Sensitivity")
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.legend(loc="lower right")
        return fig


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
        img = skut.img_as_float(img)
        if lowest_pixel is None:
            x_min, y_min = np.unravel_index(np.argmin(img_new, axis=None), img_new.shape)
            lowest_pixel = img_new[x_min, y_min]
        elif lowest_pixel == "otsu":
            lowest_pixel = skfl.threshold_otsu(img_new)
        if highest_pixel is None:
            x_max, y_max = np.unravel_index(np.argmax(img_new, axis=None), img_new.shape)
            highest_pixel = img_new[x_max, y_max]

        scaling_factor = (upper_limit - lower_limit)/(highest_pixel - lowest_pixel)
        for row_num, _ in enumerate(img_new):
            for col_num, _ in enumerate(img_new[row_num]):
                img_new[row_num, col_num] = (img_new[row_num, col_num] - lowest_pixel) * scaling_factor + lower_limit

        if flatten:
            return img_new.flatten()

        return skut.img_as_float(img_new)

    def get_regions(self, filepath_img, n_rows=2, n_cols=2):
        """Segregates image into boxes or regions

        Args:
            filepath_img (str): Filepath of the image (grayscale).
            n_rows (int, optional): Number of rows. Defaults to 2.
            n_cols (int, optional): Number of columns. Defaults to 2.
        """
        img = skio.imread(filepath_img)
        # If img is not grayscale, convert to grayscale
        if img.ndim == 3:
            if img.shape[2] == 4:
                img = skcol.rgb2gray(skcol.rgba2rgb(img))
            elif img.shape[2] == 3:
                img = skcol.rgb2gray(img)
        img = skut.img_as_float(img)
        width, height = img.shape
        unit_x = height / n_rows
        unit_y = width / n_cols
        list_x = [int(i * unit_x) for i in range(0, n_rows + 1)][1:-1]
        list_y = [int(i * unit_y) for i in range(0, n_cols + 1)][1:-1]

        return list_x, list_y, img, width, height

    def draw_regions(self, filepath_img, n_rows=2, n_cols=2):
        """Draws lines on the image to segregate regions

        Args:
            filepath_img (str): Filepath of the image (grayscale).
            n_rows (int, optional): Number of rows. Defaults to 2.
            n_cols (int, optional): Number of columns. Defaults to 2.
        """
        list_x, list_y, img, width, height = self.get_regions(filepath_img, n_rows, n_cols)        
        
        # Horizontal lines
        for x in list_x:
            img[skdr.line(x, 0, x, width - 1)] = 1
        
        # Vertical lines
        for y in list_y:
            img[skdr.line(0, y, height - 1, y)] = 1

        return img

    def crop_img_by_region(self, filepath_img, n_rows, n_cols, range_rows, range_cols):
        """Crops image by regions.

        Args:
            filepath_img (str): Filepath of the image (grayscale).
            n_rows (int): number of rows.
            n_cols (int): number of columns
            range_rows ([start_row, end_row]): Range of rows to crop.
            range_cols ([start_col, end_col]): Range of columns to crop.
        """
        list_x, list_y, img, width, height = self.get_regions(filepath_img, n_rows, n_cols)
        list_x = [0] + list_x + [height]
        list_y = [0] + list_y + [width]
        start_row, end_row = range_rows
        start_col, end_col = range_cols

        return img[list_x[start_row]: list_x[end_row], list_y[start_col]: list_y[end_col]]

    def transform_img_to_size(self, img, new_img_dims, to_rgb=False):
        """Transforms image to given dimensions

        Args:
            img (np.ndarray): np array of image.
            new_img_dims ([new_width, new_height]): Dimensions of the new image.
            to_rgb (bool): Whether to convert image to RGB. Defaults to False.
        """
        width, height = img.shape
        req_width, req_height = new_img_dims
        scaling_factor = min(req_height / height, req_width / width)
        img_new = sktr.rescale(img, scaling_factor)
        new_width, new_height = img_new.shape
        if req_width - new_width == 0:
            rem_height = req_height - new_height
            if rem_height % 2 == 1:
                padding_top, padding_bottom = (rem_height // 2, rem_height // 2 + 1)
            else:
                padding_top, padding_bottom = (rem_height // 2, rem_height // 2)
            img_new = np.hstack((np.zeros((req_width, padding_top)), img_new, np.zeros((req_width, padding_bottom))))
        elif req_height - new_height == 0:
            rem_width = req_width - new_width
            if rem_width % 2 == 1:
                padding_left, padding_right = (rem_width // 2, rem_width // 2 + 1)
            else:
                padding_left, padding_right = (rem_width // 2, rem_width // 2)
            img_new = np.vstack((np.zeros((padding_left, req_height)), img_new, np.zeros((padding_right, req_height))))
        if to_rgb:
            img_new = np.expand_dims(img_new, -1).repeat(3, axis=-1)
        return img_new

    def plot_roc(self, y, yPred, title, filepath_save):
        """Generate ROC curve

        Args:
            y (np.ndarray): actual y values.
            yPred (np.ndarray): predicted probabilities of positive prediction.
            title (str): title of the plot.
            filepath_save (str): Where to save the figure.
        """
        fig = plt.figure(figsize=(10, 10))
        fpr, tpr, thresholds = le_me.roc_curve(y, yPred, pos_label=True)
        roc_auc = le_me.auc(fpr, tpr)
        plt.plot(fpr, tpr, "-", label=f"AUC = {round(roc_auc, 3)}", linewidth=4)
        plt.xlabel("1 - Specificity")
        plt.ylabel("Sensitivity")
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.title(title)
        plt.legend(loc="lower left")
        plt.savefig(filepath_save)

        return fig, fpr, tpr, thresholds

    def plot_confusion_matrix(self, conf_matrix, title, filepath_save):
        """Plot confusion matrix

        Args:
            conf_matrix (np.ndarray): Confusion matrix.
            title (str): title of the plot.
            filepath_save (str): Where to save the figure.
        """
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(x=j, y=i,s=conf_matrix[i, j], va="center", ha="center", size="xx-large")
        
        plt.xlabel("Predictions", fontsize=18)
        plt.ylabel("Actuals", fontsize=18)
        plt.title(title, fontsize=18)
        plt.savefig(filepath_save)
        return fig