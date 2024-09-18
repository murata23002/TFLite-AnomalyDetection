import argparse
import pickle
import time

import cv2
import tensorflow.lite as tflite
from annoy import AnnoyIndex


# Load saved features from a pickle file
def load_saved_features(pickle_file):
    with open(pickle_file, "rb") as f:
        features_dict = pickle.load(f)
    return features_dict


# Load a TFLite model
def load_tflite_model(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


# OpenCV image preprocessing (resize and reshape)
def preprocess_image_cv2(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    img = img.astype("float32") / 255.0
    img = img.reshape(1, target_size[0], target_size[1], 3)  # Add batch dimension
    return img


# Extract features using TFLite model
def extract_image_features_tflite(image_path, interpreter, layer_depth):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    all_layer_outputs = []

    # Preprocess and set the input tensor
    img = preprocess_image_cv2(image_path)

    start_time = time.time()  # Start timing
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    # Retrieve features for each layer depth
    for i in range(layer_depth):
        output_data = interpreter.get_tensor(output_details[i]["index"])
        all_layer_outputs.append(output_data.flatten())  # Flatten the output to 1D

    end_time = time.time()  # End timing
    print(f"Feature extraction took {end_time - start_time:.4f} seconds")

    return all_layer_outputs


# Build separate Annoy indices for each stage's features
def build_annoy_index_from_multi_stage_features(features_dict, num_trees=1000):
    desired_shapes = [
        features_dict[i]["features"].shape[1] for i in range(len(features_dict))
    ]
    annoy_indices = {
        i: AnnoyIndex(dim, "angular") for i, dim in enumerate(desired_shapes)
    }
    image_paths = features_dict[0]["image_paths"]  # Ensure image paths are consistent

    # Add each stage's features to the respective Annoy index
    for stage, entry in enumerate(features_dict):
        stage_features = entry["features"]

        # Add features to the corresponding Annoy index for each stage
        for idx, features in enumerate(stage_features):
            annoy_indices[stage].add_item(idx, features)

    # Build Annoy indices for all stages
    for stage in annoy_indices:
        annoy_indices[stage].build(num_trees)

    return annoy_indices, image_paths  # Return indices and image paths


# Find similar images using the Annoy index for a specific stage
def find_similar_images(query_feature, annoy_indices, img_paths, num_results=5):
    similar_images = []
    distances = []
    for i_lay, query in enumerate(query_feature):
        start_time = time.time()  # Start timing
        nearest_indices = annoy_indices[i_lay].get_nns_by_vector(
            query, num_results, include_distances=True
        )
        end_time = time.time()  # End timing

        print(f"Annoy search took {end_time - start_time:.4f} seconds")

        similar_images.append(
            [img_paths[i] for i in nearest_indices[0]]
        )  # Extract image paths
        distances.append(nearest_indices[1])

    return similar_images, distances


# Main logic
def main(args):
    # Load saved features
    features_dict = load_saved_features(args.features_pickle)

    # Build Annoy indices for each stage
    annoy_indices, img_paths = build_annoy_index_from_multi_stage_features(
        features_dict
    )

    # Load TFLite model
    interpreter = load_tflite_model(args.tflite_model)

    # Extract query image features using TFLite
    query_features = extract_image_features_tflite(
        args.query_image, interpreter, len(features_dict)
    )

    # Perform Annoy search
    annoy_images, annoy_distances = find_similar_images(
        query_features, annoy_indices, img_paths
    )

    # Output similar images
    print(f"Similar images to {args.query_image} using Annoy:")
    for i, (img, dist) in enumerate(zip(annoy_images, annoy_distances), start=1):
        print(f"{i}. Image: {img}")
        print(f"   Distance: {dist}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image search using TFLite and Annoy")
    parser.add_argument(
        "--features_pickle",
        type=str,
        required=True,
        help="Path to saved features pickle file",
    )
    parser.add_argument(
        "--tflite_model", type=str, required=True, help="Path to TFLite model"
    )
    parser.add_argument(
        "--query_image", type=str, required=True, help="Path to query image"
    )
    parser.add_argument(
        "--num_results", type=int, default=5, help="Number of results to retrieve"
    )

    args = parser.parse_args()

    main(args)
