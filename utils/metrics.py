import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, classification_report,  roc_curve, auc, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def print_classification_report(df, label_col='label', pred_col='y_pred'):
    print("Classification Report:")
    print(classification_report(df[label_col], df[pred_col]))


def plot_confusion_matrix(df, label_col='label', pred_col='y_pred', title='Confusion Matrix'):
    cm = confusion_matrix(df[label_col], df[pred_col])

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=['Not Sexist (0)', 'Sexist (1)'], yticklabels=['Not Sexist (0)', 'Sexist (1)'])
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    plt.title(title)
    plt.show()

def plot_boxplot_predicted_probabilities(df, label_col='label', prob_col='prob_1', title='Boxplot of Predicted Probabilities by True Label'):
    # the output of this does not make sense to me yet. Has to be reviewed!!
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[label_col], y=df[prob_col])
    plt.xlabel('True Class Label')
    plt.ylabel('Predicted Probability for Class 1')
    plt.title(title)
    plt.show()

def plot_roc_curve_with_optimal_point(df, label_col='label', prob_col='prob_1', title='ROC Curve with Optimal Point'):
    y_true = df[label_col]
    y_prob = df[prob_col]
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    optimal_tpr = tpr[optimal_idx]

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(optimal_fpr, optimal_tpr, marker='o', color='red', label=f'Optimal Point (Threshold = {optimal_threshold:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show()

def plot_balanced_accuracy_vs_cutoff(df, label_col='label', prob_col='prob_1', title='Balanced Accuracy vs Cutoff'):
    y_true = df[label_col]
    cutoffs = np.linspace(0, 1, 100)
    balanced_accuracies = []

    for cutoff in cutoffs:
        y_pred = (df[prob_col] >= cutoff).astype(int)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        balanced_accuracies.append(balanced_acc)

    optimal_idx = np.argmax(balanced_accuracies)
    optimal_cutoff = cutoffs[optimal_idx]

    plt.figure(figsize=(8, 6))
    plt.plot(cutoffs, balanced_accuracies, color='blue', lw=2)
    plt.axvline(x=optimal_cutoff, color='red', linestyle='--', label=f'Optimal Cutoff = {optimal_cutoff:.2f}')
    plt.xlabel('Cutoff')
    plt.ylabel('Balanced Accuracy')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show()

def plot_calibration_curve(df, label_col='label', prob_col='prob_1', n_bins=10, title='Calibration Curve'):
    y_true = df[label_col]
    y_prob = df[prob_col]
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=n_bins)

    plt.figure(figsize=(10, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, marker='o', linestyle='-', label='Calibration Curve')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_probability_distribution(df, prob_col_1='prob_1', prob_col_0='prob_0'):
    plt.figure(figsize=(8, 6))
    plt.hist(df[prob_col_1], bins=20, alpha=0.7, color='b')
    plt.xlabel('Probability of being Sexist (Class 1)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Probabilities for Class 1')
    plt.show()
    # should obviously be exactly inverted, can be used as test
    # plt.figure(figsize=(8, 6))
    # plt.hist(df[prob_col_0], bins=20, alpha=0.7, color='g')
    # plt.xlabel('Probability of not being Sexist (Class 0)')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of Predicted Probabilities for Class 0')
    # plt.show()



if __name__ == '__main__':
    # Testing the functions:
    df_output = pd.read_csv("../data_submission/dl_predictions.csv")
    print_classification_report(df_output)
    plot_confusion_matrix(df_output)
    plot_probability_distribution(df_output)
    plot_boxplot_predicted_probabilities(df_output)
    plot_roc_curve_with_optimal_point(df_output)
    plot_calibration_curve(df_output)
    plot_balanced_accuracy_vs_cutoff(df_output)