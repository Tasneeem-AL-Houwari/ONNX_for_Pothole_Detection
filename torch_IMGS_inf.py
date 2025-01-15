from ultralytics import YOLO
import os
model = YOLO("C:/Users/user/Desktop/project/runs/detect/Yolov11_v1/weights/last.pt") # model path


images_path = "C:/Users/user/Desktop/project/whole_dataset/test/images"

for i in os.listdir(images_path):
    image = images_path+"/"+i
    model.predict(image,save=True,conf=0.5)