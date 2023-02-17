import configparser
from ultralytics import YOLO
from roboflow import Roboflow

config = configparser.ConfigParser()
config.read('config.ini')
apik = config.get('roboflow', 'key')

# Download the model

rf = Roboflow(api_key=apik)
project = rf.workspace("daniil-yarmov").project("all-dogs")
dataset = project.version(1).download("yolov8")


# Train the model
# yolo task=detect mode=train model=yolov8n.pt data=C:\Users\gamin\PycharmProjects\Yolodog\data.yaml epochs=3 imgsz=416  from ultralytics import YOLO

# Test model
model = YOLO('runs/detect/train/weights/best.pt')
model.predict(
   source='dog.jpg',
   conf=0.6
)
# yolo task=detect mode=predict model="C:\Users\gamin\PycharmProjects\Yolodog\runs\detect\train\weights\best.pt" source="dog.jpg"


