import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Load the data from a CSV file
csv_file = "cry1realvariations (1).csv"  # Update the path to your CSV file
df = pd.read_csv(csv_file)

# Assuming the CSV file has columns 'Mutation ID' and 'Allele Frequency'
data_id_array = df['_displayName'].values[:750].reshape(-1, 1)  # Get the first 4000 Mutation IDs
array_data = df['AF'].values[:750].reshape(-1, 1)  # Get the first 4000 Allele Frequencies

# Initialize and fit the Isolation Forest model
clf = IsolationForest(contamination=0.01, random_state=42)
clf.fit(array_data)

# Get anomaly scores and predictions
anomaly_scores = clf.decision_function(array_data)  # Negative scores represent outliers (anomalous)
predictions = clf.predict(array_data)  # -1 for anomaly, 1 for normal

# Invert the anomaly scores (as lower scores indicate anomalies, we invert so higher scores are more "normal")
inverted_anomaly_scores = -anomaly_scores

# Filter out NaN values and prepare the data for plotting
valid_indices = ~np.isnan(array_data)
valid_indices = valid_indices.flatten() 
data_id_array = data_id_array[valid_indices]
array_data = array_data[valid_indices]
inverted_anomaly_scores = inverted_anomaly_scores[valid_indices]
anomaly_scores = anomaly_scores[valid_indices]
predictions = predictions[valid_indices]

# Create a DataFrame for plotting
df_plot = pd.DataFrame({
    'Mutation ID': data_id_array.flatten(),
    'Anomaly Score': anomaly_scores,
    'Anomaly Prediction': predictions
})

# Plotly scatter plot for visualizing anomalies
fig = px.scatter(df_plot, x='Mutation ID', color='Anomaly Prediction',
                 color_continuous_scale='RdBu', title="Isolation Forest Anomaly Detection",
                 labels={'Mutation ID': 'Mutation ID', 'Inverted Anomaly Score': 'Inverted Anomaly Score'},
                 opacity=0.6)
fig.update_layout(
    width=1000,  # Set the width of the plot (you can adjust this number)
    height=800, 
    title_font=dict(size=24),  # Increase the font size for the title
    xaxis_title_font=dict(size=18),  # Increase the font size for the x-axis label
    yaxis_title_font=dict(size=18), # Set the height of the plot (you can adjust this number)
)
fig.show()

# Save the anomaly scores to a CSV file for further analysis
output_file = "E:/datasets/processeddata/750_ANOMALY_SCORES.csv"
df_plot.to_csv(output_file, index=False)

print(f"Anomaly scores and predictions saved to {output_file}")
