# Clothing Classification using Artificial Neural Network

This project utilizes TensorFlow and Keras to implement an Artificial Neural Network for classifying clothing items based on the Fashion MNIST dataset.

## Libraries Used
- `tensorflow`: TensorFlow library for building and training neural networks.
- `numpy`: NumPy for numerical computing.
- `matplotlib.pyplot`: Matplotlib for plotting images.

## Dataset
- The Fashion MNIST dataset is used for training and testing the model.

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
```

# Load Fashion MNIST dataset
clothes = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = clothes.load_data()
