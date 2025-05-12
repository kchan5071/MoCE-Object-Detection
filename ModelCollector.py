from ResNet.faster_r_cnn import ResNetModel
from Yolov5.Yolov5Detection import YoloModel
from VisionTransformer.detr_utils import make_pred, WillFlow, clean_img

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

import cv2
import os
from torchvision import transforms

def get_test_image():
    # Load the image
    image_path = "Test_image.png"
    image = cv2.imread(image_path)
    return image

def get_test_image_path():
    # Load the image
    image_path = "Test_image.png"
    return image_path

def test_yolo_model(model):

    # Get the test image
    image = get_test_image()
    
    # Run detection
    results = yolo_model.detect_in_image(image)
    
    # Print results
    return(results)

def test_resnet_model(model):

    # Get the test image
    image = get_test_image_path()
    
    # Print results
    return(resnet_model.detect_in_image(image))

def test_detr_model(model):
    model.load_state_dict(
            torch.load("VisionTransformer/best.pth", map_location=torch.device('cpu'), weights_only=True)
        )

    image = cv2.imread("Test_image.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return(make_pred(model, clean_img(image)))
    # Print results

def test_all_models(yolo_model, resnet_model, detr_model):
    results = np.zeros((3, 5))

    # Test YOLO model
    
    results[0] = np.array(test_yolo_model(yolo_model).split(", "))

    # Test ResNet model
    results[1] = np.array(test_resnet_model(resnet_model).split(", "))

    # Test DETR model
    results[2] = np.array(test_detr_model(detr_model).split(", "))

    return results
    

def initialize_models():
    # Initialize all models
    yolo_model = YoloModel('Yolov5/best.pt')
    resnet_model = ResNetModel('ResNet/best.pth')
    detr_model = WillFlow()
    return yolo_model, resnet_model, detr_model

def draw_boxes_on_image(image, results, resolution):
    for result in results:
        # Convert cx, cy, w, h to x_min, y_min, x_max, y_max
        cx, cy, w, h = result[1:5]
        x_min = int((cx - w / 2) * resolution)
        y_min = int((cy - h / 2) * resolution)
        x_max = int((cx + w / 2) * resolution)
        y_max = int((cy + h / 2) * resolution)

        # Draw the bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    return image

if __name__ == "__main__":
    # Initialize models
    yolo_model, resnet_model, detr_model = initialize_models()
    # Test all models
    results = test_all_models(yolo_model, resnet_model, detr_model)
    
    print("Results from all models:")
    for i, result in enumerate(results):
        print(f"Model {i+1} results: {result}")

    resolution = 640

    # Draw boxes on the image
    image = get_test_image()
    for i, result in enumerate(results):
        # Convert result string to a list of floats
        image = draw_boxes_on_image(image, [result], resolution)

    # Show the image
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
