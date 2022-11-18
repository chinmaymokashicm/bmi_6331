"""
Pipeline 2:
1. Crop region of interest
2. Perform contrast stretching
3. Convert image to ResNet50 format
4. Save image
5. Create feature matrix
6. Run Logistic Regression
7. Run evaluation
8. Draw ROC curve
"""

import func.task1 as t1

import pandas as pd
import numpy as np

from tqdm import tqdm

import skimage.io as skio
import skimage.transform as sktr
import skimage.color as skcol
import sklearn.model_selection as le_ms
import sklearn.preprocessing as le_pr
import sklearn.linear_model as le_lm
import sklearn.metrics as le_me
import pickle

import tensorflow.keras.applications.resnet50 as k50

import yaml, os

import logging

from datetime import timezone
import datetime

dt = datetime.datetime.now(timezone.utc)
utc_time = dt.replace(tzinfo=timezone.utc)
utc_timestamp = utc_time.timestamp()

logging.basicConfig(format="%(process)d-%(asctime)s-%(levelname)s-%(message)s",  handlers=[logging.FileHandler(f"{utc_timestamp}.log"), logging.StreamHandler()], level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")

logging.info("Started the script")

with open("func/params.yaml", "r") as f:
    dict_params = yaml.full_load(f)["task1"]

r50 = k50.ResNet50(include_top=False, pooling="avg")
logging.info(r50)

df_info = pd.read_csv("dataset/dataInfo.csv", index_col=False)
df_info["filepath"] = df_info["FullFileName"].apply(lambda path: os.path.join("dataset", path))

code_obj = t1.Task1()

folderpath_preprocess_train = os.path.join("dataset", "preprocess", "train")
folderpath_preprocess_test = os.path.join("dataset", "preprocess", "test")

folderpath_save = os.path.join("dataset", "results")

for folderpath in [folderpath_preprocess_train, folderpath_preprocess_test]:
    if os.path.exists(folderpath):
        continue
    os.makedirs(folderpath)

list_img_train = []
list_y_train = []
list_img_test = []
list_y_test = []

n_values = len(df_info.index)
i = 1

for filepath_img, train, cardiomegaly in tqdm(df_info[["filepath", "Train", "Cardiomegaly"]].values.tolist()):
    try:
        folderpath = folderpath_preprocess_train if train == 1 else folderpath_preprocess_test
        if not os.path.exists(filepath_img):
            continue
        img = code_obj.crop_img_by_region(filepath_img=filepath_img, **(dict_params["crop"]))
        img = code_obj.contrast_stretch(img=img)
        img = code_obj.transform_img_to_size(img=img, **(dict_params["transfer_learning"]["ResNet50"]))
        img *= 255
        img = img.astype(np.uint8)
        skio.imsave(fname=os.path.join(folderpath, os.path.basename(filepath_img)), arr=img)
        if train == 1:
            list_img_train.append(img)
            list_y_train.append(cardiomegaly)
        else:
            list_img_test.append(img)
            list_y_test.append(cardiomegaly)
        # logging.info(f"Pre-processing: {i}/{n_values} {filepath_img}")
    except Exception as e:
        logging.error(e, exc_info=True)
    i += 1

logging.info("Completed pre-processing of images.")

trainX = r50.predict(k50.preprocess_input(np.array(list_img_train)))
trainY =np.array(list_y_train)
logging.info("Generated final form of training dataset for machine learning.")
testX = r50.predict(k50.preprocess_input(np.array(list_img_test)))
testY = np.array(list_y_test)

logging.info("Generated final form of dataset for machine learning.")

mod1 = le_lm.LogisticRegression(penalty="l1", C=0.5, solver="liblinear")
mod1.fit(trainX, trainY)

logging.info("Trained model.")

predY = mod1.predict_proba(testX)[:, 1]

fig, fpr, tpr, thresholds = code_obj.plot_roc(testY, predY, "Logistic Regression | ResNet50", os.path.join(folderpath_save, f"{utc_timestamp}_roc.png"))

logging.info("Plotted ROC curve.")

list_diff = sorted([[tp, fp, threshold, tp - fp] for fp, tp, threshold in zip(fpr, tpr, thresholds)], reverse=True, key=lambda item: item[-1])
threshold = list_diff[0][2]
yPredAbsolute = predY > threshold

logging.info(f"Calculated threshold: {threshold}")

conf_matrix = le_me.confusion_matrix(testY, yPredAbsolute)

code_obj.plot_confusion_matrix(conf_matrix, f"Confusion matrix: Threshold: {round(threshold, 2)}", os.path.join(folderpath_save, f"{utc_timestamp}_confusion_matrix.png"))

logging.info("Plotted confusion matrix.")

with open(os.path.join(folderpath_save, f"{utc_timestamp}.pkl"), "wb") as f:
    pickle.dump(mod1, f)

logging.info("Saved model.")

np.save(os.path.join(folderpath_save, f"{utc_timestamp}_fpr"), fpr)
np.save(os.path.join(folderpath_save, f"{utc_timestamp}_tpr"), tpr)
np.save(os.path.join(folderpath_save, f"{utc_timestamp}_thresholds"), thresholds)

logging.info("Saved numpy arrays.")

dict_summary = {
    "model":
    {
        "type": "LogisticRegression",
        "penalty": "l1",
        "C": 0.5,
        "solver": "liblinear"
    },
    "transfer_learning": "ResNet50"
}
with open(os.path.join(folderpath_save, f"{utc_timestamp}.yaml"), "w") as f:
    yaml.dump(dict_summary, f)