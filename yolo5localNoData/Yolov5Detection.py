import os
import cv2
import torch
from ultralytics import YOLO


class YOLOv5Trainer:
    def __init__(self):
        self.initialized = False

    def setup(self):
        if not self.initialized:
            if not os.path.exists('yolov5'):
                os.system('git clone https://github.com/ultralytics/yolov5') # clone repository
            os.system('python -m pip install --upgrade pip')
            os.system('pip install -r yolov5/requirements.txt')
            os.system('pip install --upgrade torch torchvision torchaudio')
            self.initalized = True
        print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

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

        
class ObjDetModel:

    def __init__(self, model_name):
        print("initializing yolo object")
        # load pretrained model
        torch.cuda.set_device(0)
        self.model_resolution = 640
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path =  model_name, device = 0)
        print("yolo init success")

    def load_new_model(self, model_name):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path = './models_folder/' + model_name, device = 0)

    def detect_in_image(self, image):
        #convert color space 
        frame_cc = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #resize image to 640 x 640
        frame_squeeze = cv2.resize(frame_cc, (self.model_resolution, self.model_resolution))

        # Run the YOLO model
        results = self.model(frame_squeeze)

        #resize results according to resolution
        x_resolution = int(image.shape[1])
        y_resoltion = int(image.shape[0])

        x_ratio = x_resolution / self.model_resolution
        y_ratio = y_resoltion / self.model_resolution

        #fix resolution
        with torch.inference_mode():
            for box in results.xyxy[0]:
                if box[5] == 0:
                    box[2] = int(x_ratio * box[2])
                    box[0] = int(x_ratio * box[0])
                    box[3] = int(y_ratio * box[3])
                    box[1] = int(y_ratio * box[1])


        return results

def main():
    trainer = YOLOv5Trainer()
    trainer.trainer()

    detector = ObjDetModel('yolov5/runs/train/yolo5_local-results/weights/best.pt')

    image = cv2.imread("test.jpg")
    results = detector.detect_in_image(image)
    print(results.pandas().xyxy[0])


if __name__ == "__main__":
    main()

