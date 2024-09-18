import os

import numpy as np
import tensorflow as tf
from PIL import Image

CLASS_NAMES = ["small"]


class MVTecDataset:
    def __init__(
        self,
        root_path="./data",
        class_name="bottle",
        is_train=True,
        resize=256,
        cropsize=224,
        save_dir="./processed_images",
    ):
        assert (
            class_name in CLASS_NAMES
        ), f"class_name: {class_name}, should be in {CLASS_NAMES}"
        self.root_path = root_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize
        self.save_dir = save_dir 
        os.makedirs(self.save_dir, exist_ok=True) 
        self.mvtec_folder_path = os.path.join(root_path, "mvtec_anomaly_detection")

        # データセットの読み込み
        self.x, self.y = self.load_dataset_folder()

        # デバッグプリント
        print(f"Loaded {len(self.x)} images from class '{self.class_name}'")

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = "train" if self.is_train else "test"
        data, gt_list = [], []

        img_dir = os.path.join(self.mvtec_folder_path, self.class_name, phase)
        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted(
                [
                    os.path.join(img_type_dir, f)
                    for f in os.listdir(img_type_dir)
                    if f.endswith(".jpg") or f.endswith(".png")
                ]
            )
            data.extend(img_fpath_list)

            if img_type == "good":
                gt_list.extend([0] * len(img_fpath_list))
            else:
                gt_list.extend([1] * len(img_fpath_list))

        assert len(data) == len(gt_list), "number of x and y should be the same"
        return list(data), list(gt_list)

    def preprocess(self, img_path, label):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(
            img, [self.resize, self.resize], method=tf.image.ResizeMethod.BILINEAR
        )
        # img = tf.image.central_crop(img, self.cropsize / self.resize)
        # img = img / 255.0
        # img = (img - tf.constant([0.485, 0.456, 0.406])) / tf.constant(
        #    [0.229, 0.224, 0.225]
        # )

        return img, label

    def get_tf_dataset(self):
        # `from_tensor_slices` を使って、画像パスとラベルからデータセットを作成
        print(f"Creating TensorFlow dataset from {len(self.x)} samples")
        dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y))

        # データセットに前処理を適用
        dataset = dataset.map(
            lambda img_path, label: self.preprocess(img_path, label),
            # num_parallel_calls=tf.data.AUTOTUNE,
        )
        return dataset
