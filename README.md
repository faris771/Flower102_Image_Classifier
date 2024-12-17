# Project Image Classifier

This project is part of **Udacity Introduction to Machine Learning with Tensorflow course with Palestine Launchpad with Google** implements an image classification model using deep learning frameworks. The goal is to classify input images into predefined categories. The notebook includes data preprocessing, model training, and evaluation processes.


## Features
- **Data Preprocessing**: Normalization and augmentation of input image data.
- **Model Building**: Convolutional Neural Networks (CNNs) for image classification.
- **Evaluation**: Training accuracy, validation accuracy, and loss metrics displayed.
- **Visualization**: Training and validation curves.

## Requirements
Install the required dependencies using:
```bash
pip install tensorflow keras matplotlib numpy
```

## Usage
Run the notebook step-by-step in Jupyter Notebook:
```bash
jupyter notebook Project_Image_Classifier_Project.ipynb
```

## Project Structure
- **Data Loading**: Load images from a dataset.
- **Model Training**: Build and train a CNN model.
- **Evaluation**: Evaluate and visualize the performance metrics.

## Results
The final model achieves a classification accuracy based on the input dataset.

## Notes
- Ensure your dataset is structured correctly (train and validation folders).
- Update paths and hyperparameters as needed.


## Part 2: Command-line Prediction Script

This part introduces a command-line script for image classification using a pre-trained TensorFlow model.

### Features
- **Preprocessing**: Resize, normalize, and prepare images for inference.
- **Prediction**: Load a saved Keras model and predict the top K classes for an image.
- **Category Mapping**: Map predicted class indices to human-readable labels using a JSON file.

### Code

```python
import argparse
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import json

# Preprocess the image for inference
def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess the image: resize, normalize, and expand dimensions.
    """
    image = Image.open(image_path)
    image = image.resize(target_size)
    image = np.asarray(image) / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Predict the top K classes
def predict(image_path, model_path, top_k=5, category_names=None):
    """
    Predict the top K classes and probabilities for an input image.
    """
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)[0]
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    top_probabilities = predictions[top_indices]
    top_classes = [str(index + 1) for index in top_indices]  # 1-based index
    if category_names:
        with open(category_names, 'r') as f:
            class_names = json.load(f)
        top_classes = [class_names.get(label, "Unknown") for label in top_classes]
    return top_probabilities, top_classes

# Command-line interface
def main():
    parser = argparse.ArgumentParser(description="Predict flower class from an image.")
    parser.add_argument('image_path', type=str, help="Path to input image")
    parser.add_argument('model_path', type=str, help="Path to saved Keras model")
    parser.add_argument('--top_k', type=int, default=5, help="Return top K most likely classes")
    parser.add_argument('--category_names', type=str, help="Path to JSON file for class label mapping")
    args = parser.parse_args()

    top_probabilities, top_classes = predict(args.image_path, args.model_path, args.top_k, args.category_names)
    print("Top Predictions:")
    for i in range(len(top_probabilities)):
        print(f"{i+1}. {top_classes[i]}: {top_probabilities[i]:.4f}")

if __name__ == "__main__":
    main()
```

### Usage
1. Save the script in a Python file, e.g., `predict.py`.
2. Run the script from the command line:
   ```bash
   python predict.py <image_path> <model_path> --top_k 5 --category_names <category_json>
   ```
   - Replace `<image_path>` with the image file path.
   - Replace `<model_path>` with the path to the trained Keras model.
   - Use `--top_k` to specify the number of top predictions.
   - Optionally, use `--category_names` to map classes to labels.

### Example
```bash
python predict.py test_image.jpg my_model.h5 --top_k 3 --category_names labels.json
```

### Requirements
- `tensorflow`
- `tensorflow_hub`
- `numpy`
- `Pillow`
