"""
Class for code related to Task 2
"""

import os, json, shutil
import pandas as pd

class Task2:
    def __init__(self):
        self.folderpath_dataset = "/Users/cmokashi/Documents/UTHealth/bmi_6331/data_challenge/task2_data"
        self.folderpath_nnunet = os.path.join(self.folderpath_dataset, "nnunet")
        self.taskname = "Hippocampus"
        self.folderpath_task001_raw = os.path.join(self.folderpath_nnunet, "nnUNet_raw_data_base", "nnUNet_raw_data", f"Task001_{self.taskname}")
        with open(os.path.join(self.folderpath_dataset, "dataset-Hippocampus-BMI6331.json"), "r") as f:
            self.dict_dataset_json = json.load(f)

    def convert_dataset2nnunet(self):
        # Edit JSON file
        self.dict_dataset_json["test_copy"] = self.dict_dataset_json["test"]
        self.dict_dataset_json["test"] = []
        for idx_dict_info, dict_info in enumerate(self.dict_dataset_json["test_copy"]):
            self.dict_dataset_json["test"].append(dict_info["image"].replace("imagesTe", "imagesTs"))
        self.dict_dataset_json.pop("test_copy", None)
        
        if not os.path.exists(self.folderpath_nnunet):
            os.makedirs(os.path.join(self.folderpath_task001_raw, "imagesTr"))
            os.makedirs(os.path.join(self.folderpath_task001_raw, "imagesTs"))
            os.makedirs(os.path.join(self.folderpath_task001_raw, "labelsTr"))
            os.makedirs(os.path.join(self.folderpath_task001_raw, "labelsTs"))

        for foldername_old, foldername_new in [("imagesTe", "imagesTs"), ("imagesTr", "imagesTr")]:
            for filename in os.listdir(os.path.join(self.folderpath_dataset, foldername_old)):
                if foldername_new.startswith("images"):
                    shutil.copy(os.path.join(self.folderpath_dataset, foldername_old, filename), os.path.join(self.folderpath_task001_raw, foldername_new, filename.split(".")[0] + "_0000." + ".".join(filename.split(".")[1:])))
                else:
                    shutil.copy(os.path.join(self.folderpath_dataset, foldername_old, filename), os.path.join(self.folderpath_task001_raw, foldername_new, filename))
        
        for filename in os.listdir(os.path.join(self.folderpath_dataset, "labels")):
            if not filename.endswith(".nii.gz"):
                continue
            if os.path.exists(os.path.join(self.folderpath_dataset, "imagesTr", filename)):
                shutil.copy(os.path.join(self.folderpath_dataset, "labels", filename), os.path.join(self.folderpath_task001_raw, "labelsTr", filename))
            elif os.path.exists(os.path.join(self.folderpath_dataset, "imagesTe", filename)):
                shutil.copy(os.path.join(self.folderpath_dataset, "labels", filename), os.path.join(self.folderpath_task001_raw, "labelsTs", filename))
            else:
                print(filename)
                raise Exception("File not in training or test data")

        shutil.copy(os.path.join(self.folderpath_dataset, "dataset-Hippocampus-BMI6331.json"), os.path.join(self.folderpath_task001_raw, "dataset.json"))
        with open(os.path.join(self.folderpath_task001_raw, "dataset.json"), "w") as f:
            f.write(json.dumps(self.dict_dataset_json, indent=4))

    def evaluate_summary(self, filepath_summary="/Users/cmokashi/Documents/UTHealth/bmi_6331/data_challenge/task2_data/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task001_Hippocampus/labelsPr/summary.json"):
            # 1: anterior
            # 2: posterior
            with open(filepath_summary, "r") as f:
                dict_summary = json.load(f)
            list_results = []
            for result in dict_summary["results"]["all"]:
                list_results.append({
                "pid": os.path.basename(result["reference"].split(".")[0].split("_")[-1]),
                "anterior_dice": round(result["1"]["Dice"], 3),
                "anterior_precision": round(result["1"]["Precision"], 3),
                "anterior_recall": round(result["1"]["Recall"], 3),
                "posterior_dice": round(result["2"]["Dice"], 3),
                "posterior_precision": round(result["2"]["Precision"], 3),
                "posterior_recall": round(result["2"]["Recall"], 3)
                })

            return pd.DataFrame(list_results)

    

if __name__ == "__main__":
    code_obj = Task2()
    code_obj.convert_dataset2nnunet()
                    