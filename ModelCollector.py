from ResNet.faster_r_cnn import ResNetModel
from Yolov5.Yolov5Detection import YoloModel
from VisionTransformer.detr_utils import make_pred, WillFlow, clean_img
from lstm import LSTMModel

import torch
import numpy as np

import cv2
import os

def get_test_image():
    # Load the image
    image_path = "Test_image.png"
    image = cv2.imread(image_path)
    return image

def get_test_image_path():
    # Load the image
    image_path = "Test_image.png"
    return image_path

def inference_on_models(yolo_model, resnet_model, detr_model, image, image_path):
    # Test YOLO model
    results = np.zeros((3, 5))

    # Test YOLO model
    results[0] = np.array(yolo_model.detect_in_image(image).split(", "))
    if results[0] is None:
        results[0] = np.array([1, 0, 0, 0, 0])

    # Test ResNet model
    results[1] = np.array(resnet_model.detect_in_image(image_path).split(", "))
    if results[1] is None:
        results[1] = np.array([1, 0, 0, 0, 0])

    # Test DETR model
    results[2] = np.array(make_pred(detr_model, clean_img(image)).split(", "))
    if results[2] is None:
        results[2] = np.array([1, 0, 0, 0, 0])

    print("shape of results: ", results.shape)

    return results

def initialize_models():
    # Initialize all models
    yolo_model = YoloModel('Yolov5/best.pt')
    resnet_model = ResNetModel('ResNet/best.pth')
    detr_model = WillFlow()
    detr_model.load_state_dict(
        torch.load("VisionTransformer/best.pth", map_location=torch.device('cpu'), weights_only=True)
    )
    return yolo_model, resnet_model, detr_model

def lstm_model(input_models, hidden_dim, layer_dim, output_dim=5, max_index=100):


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
    results = inference_on_models(yolo_model, 
                                  resnet_model, 
                                  detr_model, 
                                  get_test_image(), 
                                  get_test_image_path())

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
