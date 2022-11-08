import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class Assign:
    def plot_roc(self, fpr, tpr, auc, prefix=""):
        """Plot ROC curve

        Args:
            fpr (float arrary): false positive rate
            tpr (float array): true positive rate
            auc (float): auc
            prefix (string): prefix to legend
        """
        fig = plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {round(auc, 3)}")
        plt.xlabel("1 - Specificity")
        plt.ylabel("Sensitivity")
        plt.legend()
        return fig

    def plot_roc_multiple(self, *lines):
        fig = plt.figure()
        for line in lines:
            for fpr, tpr, auc, prefix in line:
                plt.plot(fpr, tpr, label=f"{prefix}AUC = {round(auc, 3)}")
        plt.xlabel("1 - Specificity")
        plt.ylabel("Sensitivity")
        plt.legend()
        return fig