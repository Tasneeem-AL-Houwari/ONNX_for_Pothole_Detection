from ultralytics import YOLO
model = YOLO("C:/Users/user/Desktop/project/runs/detect/Yolov11_v1/weights/last.pt")

metrics = model.val(data="C:/Users/user/Desktop/project/whole_dataset/data.yaml",split='test')
print(metrics.box.map)
