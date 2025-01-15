from ultralytics import YOLO
model = YOLO("C:/Users/WB GAMING/Desktop/pothole/runs/detect/train2/weights/last.pt")  # yolov8n is efficient for limited resources

train = model.train(
    data="C:/Users/WB GAMING/Desktop/pothole/whole_dataset/data.yaml",     # Path to the dataset configuration file
    epochs=200,              # Number of epochs to train for
    imgsz=640,               # Size of input images as integer
    patience=0,              # Epochs to wait for no observable improvement for early stopping of training
    batch=32,                # Number of images per batch
    optimizer='Adam',        # Optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
    lr0=0.0001,              # Initial learning rate 
    lrf=0.01,                # Final learning rate (lr0 * lrf)
    dropout=0.25,            # Use dropout regularization (reduces overfitting)
    device=0,                # Device to run on, i.e. cuda device=0 
    seed=42,                 # Random seed for reproducibility
    workers=0,               # Number of data-loading workers (0 for serial loading)             
)

