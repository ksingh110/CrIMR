import tensorflow as tf
import apply_umap_on_data from app.py

model = tf.keras.models.load_model("/Users/krishaysingh/Downloads/6000_5_if_new_best_model.keras")
model.predict()