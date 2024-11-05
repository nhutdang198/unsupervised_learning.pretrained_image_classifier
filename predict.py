# TODO: Make all necessary imports.
import tensorflow as tf
import tf_keras
from PIL import Image
import json
import numpy as np
import argparse
import tensorflow_hub as hub

IMG_SIZE = 224

def load_class_names(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)


# TODO: Create the process_image function
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image_path)
    image = np.asarray(im)
    # Resize the image
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = np.expand_dims(image, axis=0)           # Add batch dimension

    # Normalize the image to [0, 1] range
    image = image / 255.0
    return image

def predict(image_path, model, top_k=5):
  processed_image = process_image(image_path)
  predictions = model.predict(processed_image)
  top_k_values, top_k_indices = tf.math.top_k(predictions[0], k=top_k)

  # Convert tensors to numpy arrays
  probs = top_k_values.numpy()
  classes = top_k_indices.numpy()
  return probs, classes

def main():
    parser = argparse.ArgumentParser(description='Predict image class using a trained model.')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('checkpoint', type=str, help='Model checkpoint to load')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top K classes to display')
    parser.add_argument('--category_names', type=str, default=None, help='Path to category names JSON file')

    args = parser.parse_args()


    custom_objects = {
        'KerasLayer': hub.KerasLayer
    }

    model = tf_keras.models.load_model('checkpoint.h5', custom_objects=custom_objects)
    
    
    # Predict the class of the input image
    probs, classes = predict(args.image_path, model, args.top_k)
    
    # Load class names if provided
    if args.category_names is not None:
        class_names = load_class_names(args.category_names)
        top_classes = [class_names[str(idx)] for idx in classes]
    else:
        class_names = load_class_names('label_map.json')
        top_classes = [class_names[str(idx)] for idx in classes]

    # Print results
    for prob, class_name in zip(probs, top_classes):
        print(f"Class: {class_name}, Probability: {prob:.4f}")

if __name__ == '__main__':
    main()
