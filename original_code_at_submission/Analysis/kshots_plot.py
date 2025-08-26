import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

def average_score(alist):
    res = []
    for t in alist:
        my_sum = t[0] * 3000 + t[1] * 400 + t[2] * 50
        my_avg = my_sum / 3450
        rounded_avg = round(my_avg , 2)
        res.append(rounded_avg)
    return res

# Read the CSV data
dataframe = pd.read_csv("/path/to/your/file.csv")

def extract_csv_data(df, model, hist, pr, lang, k, fb):
    conditions = []

    if model != "_":
        conditions.append(df["model"] == model)
    if hist != "_":
        conditions.append(df["chat history"] == hist)
    if pr != "_":
        conditions.append(df["Simple vs Basic"] == pr)
    if lang != "_":
        conditions.append(df["Language"] == lang)
    if k != "_":
        conditions.append(df["k-shot"] == k)
    if fb != "_":
        conditions.append(df["feedback"] == fb)

    if conditions:
        combined_condition = conditions[0]
        for condition in conditions[1:]:
            combined_condition &= condition
    else:
        combined_condition = True  

    filtered_df = df.loc[combined_condition]
    return list(filtered_df["overall"])

def projection(alist, p):
    pattern = r"\((\d+\.\d+), (\d+\.\d+), (\d+\.\d+)\)"
    for i, item in enumerate(alist):
        matches = re.findall(pattern, item)
        alist[i] = float(matches[0][p])

# Models to be used
models = ["modelA", "modelB", "modelC", "modelD"]

data = []

for model in models:
    if model == "modelC" or model == "modelA":
        config = [model, "yes", "simple", "French", "_", "yes"]
    elif model == "modelD" or model == "modelB":
        config = [model, "yes", "simple", "English", "_", "yes"]
    else:
        raise ValueError("bad config")
    model_data = extract_csv_data(dataframe, *config)

    projection(model_data, p=2)
    data.append(model_data)

print(data)

# X-axis positions
x_positions = [0, 3, 5, 7, 10]

# Create the plot
plt.figure(figsize=(12, 7))

# Define colors and labels
colors = ["red", "blue", "green", "orange"]
model_labels = ["Model A", "Model B", "Model C", "Model D"]

# Plot each line
for i, y_values in enumerate(data):
    plt.plot(x_positions, y_values, marker='o', markersize=8, linestyle='-', linewidth=2, color=colors[i], label=model_labels[i])

# Customize x-axis
plt.xticks([0, 3, 5, 7, 10], ['0-shot', '3-shot', '5-shot', '7-shot', '10-shot'], fontsize=24)
plt.xlim(-1, 11)

# Labels and title
plt.ylabel('Accuracy (CM)', fontsize=16)
plt.legend(fontsize=12, loc='upper left')
plt.grid(color='gray', linestyle='--', linewidth=0.7, alpha=0.7)

# Adjust layout and save figure
plt.tight_layout()
plt.savefig('accuracy_plot.png', dpi=300)

# Show the plot
plt.show()
