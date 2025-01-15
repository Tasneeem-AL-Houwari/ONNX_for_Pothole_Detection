
from ultralytics import YOLO

model = YOLO("C:/Users/WB GAMING/Desktop/pothole/runs/detect/yolov9_v1/weights/last.pt")  
model.export(format="onnx")