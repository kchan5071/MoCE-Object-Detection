import yaml
import os
import numpy as np
import cv2

import ModelCollector
import LSTMModel

YAML_FILE = 'pt.yaml'

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def get_image_directory(yaml_file):
    data = load_yaml(yaml_file)
    image_dir = data.get('Image_Dir', None)
    if image_dir is None:
        raise ValueError("Image directory not found in YAML file.")
    return image_dir

def get_label_directory(yaml_file):
    data = load_yaml(yaml_file)
    label_dir = data.get('Label_Dir', None)
    if label_dir is None:
        raise ValueError("Label directory not found in YAML file.")
    return label_dir

def read_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")
    return image

def read_label(label_path):
    with open(label_path, 'r') as file:
        label_data = file.readlines()
    labels = [line.strip() for line in label_data]
    return labels

def get_image_and_label(image_dir, label_dir, index):
    base_name = "screenshot_"
    image_suffix = ".png"
    label_suffix = ".txt"
    image_path = os.path.join(image_dir, base_name + f'{index:04}' + image_suffix)
    label_path = os.path.join(label_dir, base_name + f'{index:04}' + label_suffix)
    image = read_image(image_path)
    label = read_label(label_path)

    return image_path, label_path, image, label



def main():
    image_dir = get_image_directory(YAML_FILE)
    label_dir = get_label_directory(YAML_FILE)

    yolo_model, resnet_model, detr_model = ModelCollector.initialize_models()
    LSTM_model = LSTMModel.LSTMModel()

    # print(get_image_and_label(image_dir['directory'], label_dir['directory'], 1))
    for i in range(1, 2):
        image_path, label_path, image, label = get_image_and_label(image_dir['directory'], label_dir['directory'], i)
        results = ModelCollector.inference_on_models(
            yolo_model,
            resnet_model,
            detr_model,
            image,
            image_path
        )



    

if __name__ == "__main__":
    main()