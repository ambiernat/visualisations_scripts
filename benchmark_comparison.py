# ==============
# 📊 AGGREGATE RESULTS
# ==============

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os
#

# inputs
import joblib
import argparse

# Parse command-line arguments for file paths
parser = argparse.ArgumentParser(description="Generate side-by-side bar plots for actual vs predicted labels.")
parser.add_argument("--preds_path", type=str, required=True, help="Path to the saved PredictionOutput (pickle file).")
parser.add_argument("--baseline_preds_path", type=str, required=True, help="Path to the saved tokenized dataset (pickle file).")
args = parser.parse_args()

# Load objects
preds_inp = joblib.load(args.preds_path)
baseline_preds_inp = joblib.load(args.baseline_preds_path)

# Create an EDA folder if it doesn't exist
def create_eda_folder():
    current_dir = os.getcwd()

    eda_folder_path = os.path.join(current_dir, 'EDA')
    if not os.path.exists(eda_folder_path):
        os.makedirs(eda_folder_path)

def data_prep(preds, baseline_preds):

  acc = preds.metrics["test_accuracy"]
  baseline_acc = baseline_preds.metrics["test_accuracy"]

  results = pd.DataFrame({
      "Model": ["Pretrained (Unfine-Tuned)", "Fine-Tuned"],
      "Accuracy": [baseline_acc, acc]
  })

return results


# ==============
# 🎨 VISUALIZE
# ==============

def bar_plots(preds, baseline_preds):
  sns.set(style="whitegrid", font_scale=1.1)
  plt.figure(figsize=(8, 5))
  sns.barplot(
      data=results,
      x="Model",
      y="Accuracy",
      hue="Model",
      palette=["#999999", "#b3cde0"]
  )
  plt.title("Model Benchmark Comparison")
  plt.ylabel("Accuracy")
  plt.ylim(0, 1)

  for i, v in enumerate(results["Accuracy"]):
      plt.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=11)

  # Adjust layout
  plt.tight_layout()


create_eda_folder()
results = data_prep(preds=preds_inp, baseline_preds=baseline_preds_inp)
print("\n--- Model Benchmark Results ---")
print(results.to_string(index=False))

file_name = 'baseline_comparison.png'
eda_folder_path = os.path.join(os.getcwd(), 'EDA')
plot_to_save = bar_plots(preds=preds_inp, baseline_preds=baseline_preds_inp)
file_path = os.path.join(eda_folder_path, file_name)
#with open(file_path, 'w', encoding = 'utf-8-sig') as f:
plt.savefig(file_path)

text_to_display = 'File name '+file_name[:-4] +'. '+'This is also saved in the EDA folder'
print(text_to_display)
