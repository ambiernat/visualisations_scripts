import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder


# inputs
import joblib
import argparse

# Parse command-line arguments for file paths
parser = argparse.ArgumentParser(description="Generate side-by-side bar plots for actual vs predicted labels.")
parser.add_argument("--preds_base_path", type=str, required=True, help="Path to the saved PredictionOutput (pickle file).")
parser.add_argument("--preds_proposed_path", type=str, required=True, help="Path to the saved tokenized dataset (pickle file).")
args = parser.parse_args()

# Load objects
preds_base_inp = joblib.load(args.preds_base_path)
preds_proposed_inp = joblib.load(args.preds_proposed_path)


# data prep
def data_prep(preds):

    # probs and class assignments

    y_true = preds_base_inp.label_ids
    y_pred = np.argmax(preds.predictions, axis=-1)

    label_names = set(preds.label_ids.tolist())

    return y_true, y_pred, label_names


# ---- Compute Class-wise Metrics ----
def prepare_report(y_true, y_pred, label_names):

  report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)
  df_report = pd.DataFrame(report).transpose().iloc[:-3]  # remove avg/total rows
  return df_report

#

# Create an EDA folder if it doesn't exist
def create_eda_folder():
    current_dir = os.getcwd()

    eda_folder_path = os.path.join(current_dir, 'EDA')
    if not os.path.exists(eda_folder_path):
        os.makedirs(eda_folder_path)




def plot_combined(cm_base, cm_proposed, df_report_base, df_report_proposed):

  # Create side-by-side subplots
  fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 6))
  sns.set(font_scale=1.2)

  # 1️⃣ Base Case Scenario: Confusion Matrix (counts)
  sns.heatmap(cm_base, annot=True, fmt="d", cmap="Blues", ax=axs[0][0])
  axs[0][0].set_title("Base Model: Confusion Matrix (Counts)")
  axs[0][0].set_xlabel("Predicted Label")
  axs[0][0].set_ylabel("True Label")
  axs[0][0].set_xticklabels(label_names, rotation=45)
  axs[0][0].set_yticklabels(label_names, rotation=0)

  # 2️⃣ Base Case Scenario: Per-Class F1 scores
  sns.barplot(
      x=df_report_base.index,
      y=df_report["f1-score"],
      palette="Set2",
      ax=axs[0][1]
  )
  axs[0][1].set_title("Base Model: Per-Class F1 Scores")
  axs[0][1].set_xlabel("Label")
  axs[0][1].set_ylabel("F1 Score")
  axs[0][1].set_ylim(0, 1)
  axs[0][1].bar_label(axs[0][1].containers[0], fmt="%.2f")

 # 1️⃣ Proposed Scenario: Confusion Matrix (counts)
  sns.heatmap(cm_proposed, annot=True, fmt="d", cmap="Blues", ax=axs[1][0])
  axs[1][0].set_title("Proposed Model: Confusion Matrix (Counts)")
  axs[1][0].set_xlabel("Predicted Label")
  axs[1][0].set_ylabel("True Label")
  axs[1][0].set_xticklabels(label_names, rotation=45)
  axs[1][0].set_yticklabels(label_names, rotation=0)

  # 2️⃣ Proposed Scenario: Per-Class F1 scores
  sns.barplot(
      x=df_report_proposed.index,
      y=df_report["f1-score"],
      palette="Set2",
      ax=axs[1][1]
  )
  axs[1][1].set_title("Proposed Model: Per-Class F1 Scores")
  axs[1][1].set_xlabel("Label")
  axs[1][1].set_ylabel("F1 Score")
  axs[1][1].set_ylim(0, 1)
  axs[1][1].bar_label(axs[1][1].containers[0], fmt="%.2f")

  # Adjust layout
  plt.tight_layout()

create_eda_folder()

y_true_base, y_pred_base, label_names_base = data_prep(preds_base_inp)
y_true_proposed, y_pred_proposed, label_names_proposed = data_prep(preds_proposed_inp)
df_report_base = prepare_report(y_true_base, y_pred_base, label_names_base)
df_report_proposed = prepare_report(y_true_proposed, y_pred_proposed, preds_proposed_inp)

plot_to_save = plot_combined(cm_base = confusion_matrix(y_true_base, y_pred_base),
                             cm_proposed = confusion_matrix(y_true_proposed, y_pred_proposed), 
                             df_report_base = df_report_base, 
                             df_report_proposed = df_report_proposed)

file_name = 'Confusion_Matrix_and_F1.png'
eda_folder_path = os.path.join(os.getcwd(), 'EDA')

all_counts()
#plt.show()

file_path = os.path.join(eda_folder_path, file_name)
#with open(file_path, 'w', encoding = 'utf-8-sig') as f:
plt.savefig(file_path)

plt.show()
text_to_display = 'File name '+file_name[:-4] +'. '+'This is also saved in the EDA folder'
print(text_to_display)
