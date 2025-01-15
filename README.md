
# YOLO Object Detection with ONNX

This project provides an implementation for object detection using YOLO models in the ONNX format. You can process images, videos, or directories of images and save the results with detections drawn. It also supports saving the detection results in TXT format for further analysis.

## Requirements

Before using the script, ensure you have the following Python packages installed:

- `onnxruntime` - For running the ONNX models.
- `opencv-python` - For image and video processing.
- `numpy` - For numerical operations.
- `argparse` - For handling command-line arguments.

To install the dependencies, you can use the following command:

```bash
pip install onnxruntime opencv-python numpy
```

## YOLO Models

The script supports multiple YOLO models, which are available in the following versions:

- **YOLOv8** (`models/yolov8.onnx`)
- **YOLOv9** (`models/yolov9.onnx`)
- **YOLOv11** (`models/yolov11.onnx`)

You need to ensure the models are available in the `models/` directory or specify the correct path in the script.

## Usage

The script supports three types of inputs: `image`, `video`, and `images`. Based on the input type, the script will process the images and videos and save the results (with or without detections drawn) to the specified output directory.

### Command Line Arguments

| Argument          | Description                                                                          |
|-------------------|--------------------------------------------------------------------------------------|
| `--type`            | Type of input: `image`, `video`, or `images`.                                     |
| `--path`            | Path to the input (image, video, or directory).                                      |
| `--model_type`    | YOLO model version: `v8`, `v9`, or `v11`.                                            |
| `--conf_thres`    | Confidence threshold for object detection (default: 0.5).                            |
| `--iou_thres`     | Intersection over Union (IoU) threshold for non-max suppression (default: 0.5).      |
| `--save_path`     | Directory where results will be saved.                                               |
| `--save_txt`      | Flag to save detection results in TXT format (only for image input).                |
| `--save_img`      | Flag to save processed images with detections drawn (only for image or directory).  |
| `--save_video`    | Flag to save the processed video with detections drawn (only for video input).      |

### Examples

#### Process a Video:

To process a video and save the result as a new video:

```bash
python inferance.py --type video --path "    " --model_type v8 --save_path output_directory --save_video
```

#### Process a Single Image:

To process a single image and save the result as an image and TXT file:

```bash
python inferance.py --type image --path "    " --model_type v8 --save_path output_directory --save_img --save_txt
```

#### Process a Directory of Images:

To process all images in a directory:

```bash
python inferance.py images path_to_directory --model_type v8 --save_path output_directory --save_img --save_txt
```

### Explanation of Results

- **Processed Video**: If you use the `--save_video` flag, the processed video will be saved in the specified directory with detections drawn.
- **Processed Images**: If you use the `--save_img` flag, each processed image will be saved in the specified directory with bounding boxes and labels drawn around detected objects.
- **TXT Results**: If you use the `--save_txt` flag, a `.txt` file will be saved for each image. Each line in the file contains the detection results: `class_id x_center y_center width height confidence`.

### File Format for TXT

Each line in the generated `.txt` file corresponds to a detected object in the image and has the following format:

```
class_id x_center y_center width height confidence
```

Where:
- `class_id`: The ID of the detected object class.
- `x_center, y_center`: The center of the bounding box (relative to the image width and height).
- `width, height`: The width and height of the bounding box (relative to the image width and height).
- `confidence`: The confidence score of the detection.

### Example Directory Structure

```
.
├── models
│   ├── yolov8.onnx
│   ├── yolov9.onnx
│   └── yolov11.onnx
├── process_video.py
└── README.md
```

## Notes

- Make sure to use the correct paths for both input files and models.
- The `--save_video` option only works for video input, and `--save_img` and `--save_txt` options only work for image and directory input.
- To stop the video processing while it's running, press the `q` key.
