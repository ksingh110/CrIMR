import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
data = np.load("E:\datasets\processeddata\cry1mutationallelefrequency.npz")
data_id = np.load("E:\datasets\processeddata\cry1mutationid.npz")
data_id_array = data_id['arr_0']
data_id_array = data_id_array[:4000]
data_id_array = data_id_array.reshape(-1, 1)
array_data = data['arr_0']
array_data = array_data[:4000]
array_data = array_data.reshape(-1, 1)
clf = IsolationForest(contamination=0.01, random_state=42)
clf.fit(array_data)
anomaly_scores = clf.decision_function(array_data)
predictions = clf.predict(array_data)
print(anomaly_scores)
print(predictions)
x = data_id_array
y = array_data
valid_indices = ~np.isnan(y)  # Filter out NaN values
x = [str(item) for item in data_id_array]
y = y[valid_indices]
df = pd.DataFrame({'Mutation ID': x, 'Allele Frequency': y, 'Anomaly Score': anomaly_scores, "Anomaly Prediction": predictions})
fig = px.scatter(df, x='Mutation ID', y='Anomaly Score', color='Anomaly Prediction',
                 color_continuous_scale='RdBu', title="Isolation Forest Anomaly Detection",
                 labels={'Mutation ID': 'Mutation ID', 'Anomaly Score': 'Anomaly Score'},
                 opacity=0.5)

# Show the plot
fig.show()

#plt.bar(predictions, y, alpha=0.5, linewidth = 1, edgecolor = 'black')
"""plt.figure(figsize=(12, 8))
plt.scatter(x, y, s = 3, alpha = 0.5)
plt.title("Isolation Forest Anomaly Detection")
plt.xlabel("Mutation ID")
plt.ylabel("Allele Frequency")
plt.xticks(rotation=90, fontsize=0.4)
plt.show()
plt.show() """
