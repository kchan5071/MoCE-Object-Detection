import os
import cv2
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionPredictor


class YOLOv5Trainer:
    def __init__(self):
        self.initialized = False

    def train(self, img_size = 640, batch_size = 16, epochs = 10):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.system(
            f'python yolov5/train.py --img {img_size} --batch {batch_size} --epochs {epochs} '
            f'--data "{script_dir}/shark/data.yaml" --weights yolov5s.pt ' 
            f'--name yolo5_local-results --cache --optimizer AdamW'
        )

    def trainer(self):
        self.setup()
        trainer = YOLOv5Trainer()
        if os.path.exists('yolov5/runs/train/yolo5_local-results/weights/best.pt'):
            print('It appears training data already exists... Retrain? y/[N]:')
            input()
            if input == 'n' or 'N':
                print('Skipping training...')
            elif input == 'y' or 'Y':
                print ('Retraining data...')
                trainer.train()
            else:
                print(
                    'Idk what that means '
                    'but skipping training anyways...'
                )

        else:
            trainer.train()

        
class YoloModel:

    def __init__(self, model_name):
        self.model_resolution = 640
        #check if model exists
        if not os.path.exists(model_name):
            print(f"Model {model_name} does not exist.")
            return
        else:
            print("Found model, initializing yolo object")


        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path =  model_name, _verbose=False)

    def detect_in_image(self, image):
        # Run the YOLO model
        results = self.model(image)

        #fix resolution
        with torch.inference_mode():
            for box in results.xyxy[0]:
                if box[5] == 0:
                    box[2] = box[2] / self.model_resolution
                    box[0] = box[0] / self.model_resolution
                    box[3] = box[3] / self.model_resolution
                    box[1] = box[1] / self.model_resolution

        # fix to 1 box
        if len(results.xyxy[0]) == 0:
            return "1, 0, 0, 0, 0"
        #fix to 1 box
        if len(results.xyxy[0]) > 1:
            results.xyxy[0] = results.xyxy[0][0:1]

        #change pandas to string with class center_x, center_y, width, height
        result_string = "1, "
        for box in results.xyxy[0]:
            if box[5] == 0:
                center_x = (box[2] + box[0]) / 2
                center_y = (box[3] + box[1]) / 2
                width = box[2] - box[0]
                height = box[3] - box[1]
                result_string += f"{(center_x)}, {(center_y)}, {(width)}, {(height)}"
        return result_string

def main():
    # trainer = YOLOv5Trainer()
    # trainer.trainer()

    detector = YoloModel('Yolov5/best.pt')

    image = cv2.imread("Test_image.png")
    results = detector.detect_in_image(image)
    print(results)


if __name__ == "__main__":
    main()

()

