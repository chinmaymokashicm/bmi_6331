"""
Class for code related to Task 1
"""
import copy, os, yaml
from tabulate import tabulate
from tqdm import tqdm as tqdm_regular
from tqdm.notebook import tqdm as tqdm_notebook
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np
import pandas as pd
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

import tensorflow.keras.applications.inception_v3 as ki3
import tensorflow.keras.applications.resnet50 as k50
import tensorflow.keras.applications.vgg16 as vgg16

import pickle

class Task1:
    def __init__(self):
        self.folderpath_root = "/Users/cmokashi/Documents/UTHealth/bmi_6331/data_challenge/"
        self.folderpath_results = os.path.join(self.folderpath_root, "dataset", "results")
        with open("func/params.yaml", "r") as f:
            self.dict_params = yaml.full_load(f)["task1"]

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
    
    def preprocess_images(self, crop="v1", transfer_learning_model="InceptionV3", tqdm_type="notebook"):
        df_info = pd.read_csv(os.path.join(self.folderpath_root, "dataset/dataInfo.csv"), index_col=False)
        df_info["filepath"] = df_info["FullFileName"].apply(lambda path: os.path.join("dataset", path))
        list_img_train = []
        list_y_train = []
        list_img_test = []
        list_y_test = []

        tqdm_module = tqdm_regular if tqdm_type == "regular" else tqdm_notebook

        for filepath_img, train, cardiomegaly in tqdm_module(df_info[["filepath", "Train", "Cardiomegaly"]].values.tolist()):
            if crop == "v1":
                img = self.crop_img_by_region(filepath_img=filepath_img, **(self.dict_params["crop"]))
            elif crop == "v2":
                img = self.crop_img_by_region(filepath_img=filepath_img, **(self.dict_params["crop_new"]))
            else:
                img = skio.imread(filepath_img)
            img = self.contrast_stretch(img=img)
            img = self.transform_img_to_size(img=img, **(self.dict_params["transfer_learning"][transfer_learning_model]))
            img *= 255
            img = img.astype(np.uint8)
            if train == 1:
                list_img_train.append(img)
                list_y_train.append(cardiomegaly)
            else:
                list_img_test.append(img)
                list_y_test.append(cardiomegaly)

        if transfer_learning_model == "InceptionV3":
            net = ki3
            tl_mod = net.InceptionV3(include_top=False, pooling="avg")
        elif transfer_learning_model == "ResNet50":
            net = k50
            tl_mod = net.ResNet50(include_top=False, pooling="avg")
        elif transfer_learning_model == "VGG16":
            net = vgg16
            tl_mod = net.VGG16(include_top=False, pooling="avg")

        trainX = tl_mod.predict(net.preprocess_input(np.array(list_img_train)))
        trainY =np.array(list_y_train)
        testX = tl_mod.predict(net.preprocess_input(np.array(list_img_test)))
        testY = np.array(list_y_test)

        return trainX, trainY, testX, testY

    def return_group_info(self, tqdm_type="notebook"):
        df_info = pd.read_csv(os.path.join(self.folderpath_root, "dataset/dataInfo.csv"), index_col=False)
        df_info["filepath"] = df_info["FullFileName"].apply(lambda path: os.path.join("dataset", path))
        list_X_train = []
        list_y_train = []
        list_X_test = []
        list_y_test = []

        tqdm_module = tqdm_regular if tqdm_type == "regular" else tqdm_notebook

        for train, cardiomegaly, age, gender, view_pos in tqdm_module(df_info[["Train", "Cardiomegaly", "PatAge", "PatGender", "ViewPos"]].values.tolist()):
            x = [age, gender, view_pos]
            y = cardiomegaly
            if train == 1:
                list_X_train.append(x)
                list_y_train.append(y)
            else:
                list_X_test.append(x)
                list_y_test.append(y)
        list_X_train = np.array(list_X_train)
        list_y_train = np.array(list_y_train)
        list_X_test = np.array(list_X_test)
        list_y_test = np.array(list_y_test)
        return list_X_train, list_y_train, list_X_test, list_y_test


    def load_plot_roc(self):
        """Load data to plot ROC curves.
        """
        plt.rc("axes", prop_cycle=(cycler("color", ["r", "g", "b", "y"])))
        fig = plt.figure(figsize=(10, 10))
        fig.suptitle("ROC curves")
        list_uids, list_uids_complete = self.get_pipelines()
        list_uids_auc = []
        for uid in list_uids_complete:
            fpr = np.load(os.path.join(self.folderpath_results, f"{uid}_fpr.npy")), 
            tpr = np.load(os.path.join(self.folderpath_results, f"{uid}_tpr.npy")), 
            thresholds = np.load(os.path.join(self.folderpath_results, f"{uid}_thresholds.npy"))
            auc = le_me.auc(fpr[0], tpr[0])
            list_uids_auc.append((uid, auc))
        for uid, _ in sorted(list_uids_auc, key=lambda l: l[1], reverse=True):
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

    def generate_model_performance_summary(self, tqdm_type="notebook"):
        df_info = pd.read_csv("dataset/dataInfo.csv", index_col=False)
        df_info["filepath"] = df_info["FullFileName"].apply(lambda path: os.path.join("dataset", path))
        trainXInceptionV3, trainYInceptionV3, testXInceptionV3, testYInceptionV3 = np.load(os.path.join(self.folderpath_root, "dataset", "_trainX_InceptionV3.npy")), np.load(os.path.join(self.folderpath_root, "dataset", "_trainY_InceptionV3.npy")), np.load(os.path.join(self.folderpath_root, "dataset", "_testX_InceptionV3.npy")), np.load(os.path.join(self.folderpath_root, "dataset", "_testY_InceptionV3.npy"))
        trainXInceptionV3_v2, trainYInceptionV3_v2, testXInceptionV3_v2, testYInceptionV3_v2 = np.load(os.path.join(self.folderpath_root, "dataset", "_trainX_InceptionV3_v2.npy")), np.load(os.path.join(self.folderpath_root, "dataset", "_trainY_InceptionV3_v2.npy")), np.load(os.path.join(self.folderpath_root, "dataset", "_testX_InceptionV3_v2.npy")), np.load(os.path.join(self.folderpath_root, "dataset", "_testY_InceptionV3_v2.npy"))
        trainXResNet50, trainYResNet50, testXResNet50, testYResNet50 = np.load(os.path.join(self.folderpath_root, "dataset", "_trainX_ResNet50.npy")), np.load(os.path.join(self.folderpath_root, "dataset", "_trainY_ResNet50.npy")), np.load(os.path.join(self.folderpath_root, "dataset", "_testX_ResNet50.npy")), np.load(os.path.join(self.folderpath_root, "dataset", "_testY_ResNet50.npy"))
        trainXResNet50_v2, trainYResNet50_v2, testXResNet50_v2, testYResNet50_v2 = np.load(os.path.join(self.folderpath_root, "dataset", "_trainX_ResNet50_v2.npy")), np.load(os.path.join(self.folderpath_root, "dataset", "_trainY_ResNet50_v2.npy")), np.load(os.path.join(self.folderpath_root, "dataset", "_testX_ResNet50_v2.npy")), np.load(os.path.join(self.folderpath_root, "dataset", "_testY_ResNet50_v2.npy"))
        trainXVGG16, trainYVGG16, testXVGG16, testYVGG16 = np.load(os.path.join(self.folderpath_root, "dataset", "_trainX_VGG16.npy")), np.load(os.path.join(self.folderpath_root, "dataset", "_trainY_VGG16.npy")), np.load(os.path.join(self.folderpath_root, "dataset", "_testX_VGG16.npy")), np.load(os.path.join(self.folderpath_root, "dataset", "_testY_VGG16.npy"))
        list_uids, list_uids_complete = self.get_pipelines()
        list_dict_rows = []
        list_df_groups = [] # Generating model-wise group (gender, age) performance 

        arr_X_train, arr_y_train, arr_X_test, arr_y_test = self.return_group_info(tqdm_type=tqdm_type)

        tqdm_module = tqdm_regular if tqdm_type == "regular" else tqdm_notebook

        for uid in tqdm_module(list_uids_complete):
            with open(os.path.join(self.folderpath_results, f"{uid}.yaml"), "r") as f:
                dict_info = yaml.full_load(f)
            
            model_type = dict_info["model"]["type"]
            transfer_learning_type = dict_info["transfer_learning"]
            version = None
            if "version" in dict_info["model"]:
                version = dict_info["model"]["version"]

            with open(os.path.join(self.folderpath_results, f"{uid}.pkl"), "rb") as f_mod:
                mod = pickle.load(f_mod)

            if transfer_learning_type == "InceptionV3":
                if version == "v2":
                    X_train, y_train, X_test, y_test = trainXInceptionV3_v2, trainYInceptionV3_v2, testXInceptionV3_v2, testYInceptionV3_v2
                else:
                    X_train, y_train, X_test, y_test = trainXInceptionV3, trainYInceptionV3, testXInceptionV3, testYInceptionV3
            elif transfer_learning_type == "ResNet50":
                if version == "v2":
                    X_train, y_train, X_test, y_test = trainXResNet50_v2, trainYResNet50_v2, testXResNet50_v2, testYResNet50_v2
                else:
                    X_train, y_train, X_test, y_test = trainXResNet50, trainYResNet50, testXResNet50, testYResNet50
            elif transfer_learning_type == "VGG16":
                X_train, y_train, X_test, y_test = trainXVGG16, trainYVGG16, testXVGG16, testYVGG16
            else:
                raise Exception(f"Could not find transfer learning type: {transfer_learning_type}")
            
            # Training Data
            y_pred_train = mod.predict_proba(X_train)[:, 1]
            fpr_train, tpr_train, thresholds_train = le_me.roc_curve(y_train, y_pred_train, pos_label=True)
            threshold_train = self.calculate_threshold(fpr_train, tpr_train, thresholds_train)
            y_pred_train_abs = y_pred_train > threshold_train

            df_train = pd.DataFrame(list(zip(y_train, y_pred_train_abs, arr_X_train[:, 0], arr_X_train[:, 1], arr_X_train[:, 2])), columns=["true", "pred", "age", "gender", "view_pos"])
            df_train["train"] = 1
            df_train["model"] = f"{transfer_learning_type}_{model_type}" if version is None else f"{transfer_learning_type}_v{version}_{model_type}"

            f1_train = le_me.f1_score(y_train, y_pred_train_abs)
            auc_train = le_me.auc(fpr_train, tpr_train)

            # Test Data
            fpr_test = np.load(os.path.join(self.folderpath_results, f"{uid}_fpr.npy"))
            tpr_test = np.load(os.path.join(self.folderpath_results, f"{uid}_tpr.npy"))
            thresholds_test = np.load(os.path.join(self.folderpath_results, f"{uid}_thresholds.npy"))
            threshold_test = self.calculate_threshold(fpr_test, tpr_test, thresholds_test)

            y_pred_test = mod.predict_proba(X_test)[:, 1]
            y_pred_test_abs = y_pred_test > threshold_test

            df_test = pd.DataFrame(list(zip(y_test, y_pred_test_abs, arr_X_test[:, 0], arr_X_test[:, 1], arr_X_train[:, 2])), columns=["true", "pred", "age", "gender", "view_pos"])
            df_test["train"] = 0
            df_test["model"] = f"{transfer_learning_type}_{model_type}" if version is None else f"{transfer_learning_type}_v{version}_{model_type}"

            df_group = pd.concat([df_train, df_test])
            list_df_groups.append(df_group)

            f1_test = le_me.f1_score(y_test, y_pred_test_abs)
            auc_test = le_me.auc(fpr_test, tpr_test)

            list_dict_rows.append({
                "uid": uid,
                "transfer_learning_type": transfer_learning_type if version is None else f"{transfer_learning_type}_v{version}",
                "model_type": model_type,
                "train": {
                    "threshold": threshold_train,
                    "f1": f1_train,
                    "auc": auc_train
                },
                "test": {
                    "threshold": threshold_test,
                    "f1": f1_test,
                    "auc": auc_test
                }
            })

        df_summary = pd.json_normalize(list_dict_rows)
        df_summary["unique_type"] = df_summary["transfer_learning_type"].astype(str) + " | " + df_summary["model_type"].astype(str)
        # ax = df_summary.plot.bar(x="unique_type", y=["train.auc", "test.auc"])
        # ax.legend(["train", "test"])

        df_group = pd.concat(list_df_groups)

        # return df_summary, df_group, ax
        return df_summary, df_group
        

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

    def per_group_performance(self, df):
        for model_name in df["model"].unique():
            print("\n")
            print(f"=================={model_name}==================")
            df_temp = df[df["model"] == model_name]
            df_temp = df_temp.join(pd.get_dummies(df_temp["cf"]))
            for category in ["view_pos", "age_binned", "gender"]:
                for train_status in [0, 1]:
                    df_temp_train = df_temp[df_temp["train"] == train_status]
                    df_temp_train = df_temp_train.groupby([category, "train"])[pd.get_dummies(df_temp["cf"]).columns.tolist()].sum().reset_index()
                    df_temp_train["precision"] = df_temp_train["TP"]/(df_temp_train["TP"] + df_temp_train["FP"])
                    df_temp_train["recall"] = df_temp_train["TP"]/(df_temp_train["TP"] + df_temp_train["FN"])
                    df_temp_train["f1"] = (2 * df_temp_train["precision"] * df_temp_train["recall"]) / (df_temp_train["precision"] + df_temp_train["recall"])
                    # df_temp["model"] = model_name
                    print(tabulate(df_temp_train, headers = "keys", tablefmt = "psql"))
                    print("\n")
            print("\n\n")

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

    def calculate_threshold(self, fpr, tpr, thresholds):
        """Calculates best threshold value for classification probabilities.

        Args:
            fpr (np.ndarray): FPR
            tpr (np.ndarray): TPR
            thresholds (np.ndarray): thresholds
        """
        list_diff = sorted([[tp, fp, threshold, tp - fp] for fp, tp, threshold in zip(fpr, tpr, thresholds)], reverse=True, key=lambda item: item[-1])
        threshold = list_diff[0][2]
        return threshold

    def divide_chunks(self, l, chunk_size):
        """Divides a list or array or iterator into chunks

        Args:
            l (iter): iterator
            chunk_size (int): size of every chunk, except the last

        Yields:
            generator: chunk
        """
        for i in range(0, len(l), chunk_size):
            yield l[i: i + chunk_size]