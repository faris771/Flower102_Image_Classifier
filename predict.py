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
    Args:
        image_path (str): Path to the input image.
        model_path (str): Path to the trained model (Keras .h5 file).
        top_k (int): Number of top predictions to return.
        category_names (str): Path to a JSON file mapping labels to class names.
    """
    
    # Load the model
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    
    # Preprocess the image
    processed_image = preprocess_image(image_path)

    # Perform prediction
    predictions = model.predict(processed_image)[0]

    # Get the top K probabilities and class indices
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    top_probabilities = predictions[top_indices]
    top_classes = [str(index + 1) for index in top_indices]  # Convert to 1-based index
    
    # Map class indices to names if a category file is provided
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

    # Run prediction
    top_probabilities, top_classes = predict(
        args.image_path, args.model_path, args.top_k, args.category_names
    )

    # Display results
    print("Top Predictions:")
    for i in range(len(top_probabilities)):
        print(f"{i+1}. {top_classes[i]}: {top_probabilities[i]:.4f}")

if __name__ == "__main__":
    main()

