import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data
data = {
    "Class": [0.0, 1.0, "accuracy", "macro avg", "weighted avg"],
    "Precision": [0.8968, 0.9989, 0.942, 0.9478, 0.9478],
    "Recall": [0.999, 0.885, 0.942, 0.942, 0.942],
    "F1-Score": [0.9451, 0.9385, 0.942, 0.9418, 0.9418],
    "Support": [1000, 1000, 0.942, 2000, 2000],
}

# Create DataFrame
df = pd.DataFrame(data)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 4))

# Hide axes
ax.axis('off')

# Create the table and set font properties
table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center', colColours=["#315a81"] * len(df.columns))

# Loop through the cells to customize the colors
for (i, j), cell in table.get_celld().items():
    if i == 0:  # Highlight first row
        cell.set_fontsize(12)
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#315a81')
        cell.set_text_props(color='white')
    elif j == 0:  # Highlight first column
        cell.set_fontsize(12)
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#f87c00')
        cell.set_text_props(color='white')
    else:
        cell.set_fontsize(12)

# Save the table as an image
plt.savefig("highlighted_table.png", bbox_inches='tight', dpi=300)

# Display the table
plt.show()
