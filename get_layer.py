import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model

input_tensor = Input(shape=(224, 224, 3))

base_model = EfficientNetB0(
    weights="imagenet", include_top=False, input_tensor=input_tensor
)
with open(f"{base_model.name}_model_summary.txt", "w") as f:
    base_model.summary(print_fn=lambda x: f.write(x + "\n"))
