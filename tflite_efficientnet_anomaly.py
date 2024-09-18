import argparse
import csv
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow.keras.applications import (
    EfficientNetB0,
    EfficientNetB1,
    EfficientNetB2,
    EfficientNetB3,
    EfficientNetB4,
)
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tqdm import tqdm

from datasets.mvtec import CLASS_NAMES, MVTecDataset


def parse_args():
    parser = argparse.ArgumentParser("MahalanobisAD")
    parser.add_argument("--model_name", type=str, default="efficientnet-b0")
    parser.add_argument("--save_path", type=str, default="./result")
    parser.add_argument(
        "--save_feat",
        default=True,
        action="store_true",
    )
    parser.add_argument(
        "--quantize",
        default=True,
        action="store_true",
        help="Enable model quantization",
    )
    return parser.parse_args()


def calulate_mean_and_cov(feats):
    lw = LedoitWolf()
    lw.fit(feats)
    cov = lw.covariance_
    inv_cov = np.linalg.inv(cov)
    mean_feat = np.mean(feats, axis=0)

    return mean_feat, inv_cov


def calculate_mahalanobis(train_feats, test_feats, layer_depth, do_save=True):
    distances = [[] for _ in range(layer_depth)]
    for layer_idx, (train_feat, test_feat) in enumerate(zip(train_feats, test_feats)):
        mean_feat, inv_cov = calulate_mean_and_cov(train_feat)
        for test_sample in test_feat:
            dist = mahalanobis(test_sample, mean_feat, inv_cov)
            distances[layer_idx].append(dist)

    return distances


def save_distances_to_file(
    distances,
    total_distances,
    test_labels,
    output_dir,
    class_name,
    file_name,
    test_file_names,
):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(
        output_dir, f"{class_name}_{file_name}_mahalanobis_distances.csv"
    )

    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["File Name", "Label", "dist"]
            + [f"Layer {i+1}" for i in range(len(distances))]
        )
        for file_name, label, toral_dist, dist in zip(
            test_file_names, test_labels, total_distances, zip(*distances)
        ):
            writer.writerow([file_name, label, toral_dist] + list(dist))

    print(f"Distances saved to {file_path}")


def plot_and_save_roc_curve(fpr, tpr, class_name, file_name, output_dir, auc_score):
    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for {class_name} score {auc_score}")
    plt.legend()

    # ROCプロットの保存
    os.makedirs(output_dir, exist_ok=True)
    plot_file_path = os.path.join(output_dir, f"{class_name}_{file_name}_roc_curve.png")
    plt.savefig(plot_file_path)
    print(f"ROC curve saved to {plot_file_path}")
    plt.close()


def evaluate_anomaly_detection(
    train_feats,
    test_feats,
    test_labels,
    test_file_names,
    output_dir,
    file_name,
    layer_depth,
    class_name,
):
    distances = calculate_mahalanobis(train_feats, test_feats, layer_depth)
    total_distances = list(map(sum, zip(*distances)))

    auc_score = roc_auc_score(test_labels, total_distances)
    print(f"ROC AUC: {auc_score}")

    save_distances_to_file(
        distances,
        total_distances,
        test_labels,
        output_dir,
        class_name,
        file_name,
        test_file_names,
    )

    fpr, tpr, _ = roc_curve(test_labels, total_distances)
    plot_and_save_roc_curve(fpr, tpr, class_name, file_name, output_dir, auc_score)


def extract_features(model, dataset, layer_depth, class_name):
    all_layer_outputs = [[] for _ in range(layer_depth)]

    for x, _ in tqdm(dataset, f"| feature extraction {class_name}|"):
        feats = model(x.numpy(), training=False)
        for i, feat in enumerate(feats):
            all_layer_outputs[i].append(feat)

    for i in range(len(all_layer_outputs)):
        all_layer_outputs[i] = np.concatenate(all_layer_outputs[i], axis=0)

    return all_layer_outputs


def extract_features_with_tflite(interpreter, dataset, layer_depth, class_name):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    all_layer_outputs = [[] for _ in range(layer_depth)]
    for x, y in tqdm(dataset, "| TFLite inference | test | %s |" % class_name):
        for i in range(x.shape[0]):  # バッチ内の各サンプルに対してループ
            input_data = np.expand_dims(x[i].numpy(), axis=0)  # バッチサイズを1にする
            interpreter.set_tensor(input_details[0]["index"], input_data)
            interpreter.invoke()

            for i, output_detail in enumerate(output_details):
                output_data = interpreter.get_tensor(output_detail["index"])
                all_layer_outputs[i].append(output_data)

    for i in range(len(all_layer_outputs)):
        all_layer_outputs[i] = np.concatenate(all_layer_outputs[i], axis=0)

    return all_layer_outputs


def convert_to_tflite(model, save_path, model_name, quantize=False):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    quantize_model = "float32"
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]  # float16量子化を適用
        quantize_model = "float16"

    tflite_model = converter.convert()
    tflite_model_path = os.path.join(
        save_path, f"{model_name}_{quantize_model}_model.tflite"
    )

    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)

    print(f"TFLite model saved to {tflite_model_path}")
    return tflite_model_path


def prepare_datasets(class_name, input_shape, batch_size=64):
    train_dataset = MVTecDataset(
        class_name=class_name,
        is_train=True,
        cropsize=input_shape[0],
        resize=input_shape[0],
    )

    test_dataset = MVTecDataset(
        class_name=class_name,
        is_train=False,
        cropsize=input_shape[0],
        resize=input_shape[0],
    )
    train_labels = train_dataset.y
    train_file_names = train_dataset.x
    test_labels = test_dataset.y
    test_file_names = test_dataset.x

    train_dataset = train_dataset.get_tf_dataset().batch(batch_size)
    test_dataset = test_dataset.get_tf_dataset().batch(batch_size)

    return (
        train_dataset,
        test_dataset,
        test_labels,
        test_file_names,
        train_labels,
        train_file_names,
    )


def build_model(input_shape, model_name="efficientnet-b0"):

    if model_name == "efficientnet-b0":
        base_model = EfficientNetB0(
            weights="imagenet", include_top=False, input_shape=input_shape
        )
        target_layer_names = [
            "stem_activation",
            "block1a_project_bn",
            "block2b_add",
            "block3b_add",
            "block4c_add",
            "block5c_add",
            "block6d_add",
            "block7a_project_bn",
            "top_activation",
        ]
    elif model_name == "efficientnet-b4":
        base_model = EfficientNetB4(
            weights="imagenet", include_top=False, input_shape=input_shape
        )
        target_layer_names = [
            "stem_activation",
            "block1b_add",
            "block2d_add",
            "block3d_add",
            "block4f_add",
            "block5f_add",
            "block6h_add",
            "block7b_add",
            "top_activation",
        ]
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    layer_outputs = [base_model.get_layer(name).output for name in target_layer_names]
    pooled_outputs = [GlobalAveragePooling2D()(output) for output in layer_outputs]
    model = Model(inputs=base_model.input, outputs=pooled_outputs)

    layer_depth = len(target_layer_names)

    return model, layer_depth, model_name


# Example saving function to include image paths
def save_features_with_paths(
    train_feats, image_paths, cls_name, model_name, save_path, is_train=True
):
    extract_name = "train" if is_train else "test"
    # Ensure the save directory exists
    os.makedirs(os.path.join(save_path, "feats"), exist_ok=True)

    # Path to save the pickle file
    train_feat_filepath = os.path.join(
        save_path, "feats", f"{extract_name}_{cls_name}_{model_name}.pkl"
    )

    # Store both image paths and their corresponding features
    dist = []
    for i, feat in enumerate(train_feats):
        dist.append(
            {
                "image_paths": image_paths,
                "features": feat,
            }
        )

    # Save the feature list with image paths to a pickle file
    with open(train_feat_filepath, "wb") as f:
        pickle.dump(dist, f)

    print(f"Features with image paths saved to {train_feat_filepath}")


def main():
    args = parse_args()

    os.makedirs(os.path.join(args.save_path, "temp"), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "feats"), exist_ok=True)

    input_shape = (224, 224, 3)
    model, layer_depth, model_name = build_model(input_shape, args.model_name)
    tflite_model_path = convert_to_tflite(
        model, args.save_path, model_name, quantize=args.quantize
    )
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    for cls_name in CLASS_NAMES:
        (
            train_dataset,
            test_dataset,
            test_labels,
            test_file_names,
            train_labels,
            train_file_names,
        ) = prepare_datasets(cls_name, input_shape)

        train_feats = extract_features(model, train_dataset, layer_depth, cls_name)
        test_feats = extract_features(model, test_dataset, layer_depth, cls_name)

        evaluate_anomaly_detection(
            train_feats=train_feats,
            test_feats=test_feats,
            test_labels=test_labels,
            test_file_names=test_file_names,
            output_dir=args.save_path,
            file_name="keras_model",
            class_name=cls_name,
            layer_depth=layer_depth,
        )

        train_tflite_feats = extract_features_with_tflite(
            interpreter, train_dataset, layer_depth, cls_name
        )

        test_tflite_feats = extract_features_with_tflite(
            interpreter, test_dataset, layer_depth, cls_name
        )

        evaluate_anomaly_detection(
            train_feats=train_tflite_feats,
            test_feats=test_tflite_feats,
            test_labels=test_labels,
            test_file_names=test_file_names,
            output_dir=args.save_path,
            file_name="tflite_model",
            class_name=cls_name,
            layer_depth=layer_depth,
        )

        if args.save_feat:
            save_features_with_paths(
                save_path=args.save_path,
                model_name=model_name,
                cls_name=cls_name,
                image_paths=train_file_names,
                train_feats=train_tflite_feats,
            )
            save_features_with_paths(
                save_path=args.save_path,
                model_name=model_name,
                cls_name=cls_name,
                image_paths=test_file_names,
                train_feats=test_tflite_feats,
                is_train=False,
            )


if __name__ == "__main__":
    main()
