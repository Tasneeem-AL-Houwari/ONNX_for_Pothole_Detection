from ultralytics import YOLO
model = YOLO("C:/Users/user/Desktop/project/runs/detect/yolov11_v1/weights/last.pt") # load your model yolov8 , v9 , v11

source = "C:/Users/user/Desktop/project/whole_dataset/test/images/potholes46.png" # The path for image/video to test on
model.predict(source, save=True, imgsz=640, conf=0.5, name="inf11") # change the name depend on tha model that you are using