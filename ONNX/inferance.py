import argparse
import os
import cv2
from YOLO.yolo import YOLO
from YOLO.utils import draw_detections  
import time
MODEL_PATHS = {
    "v8": "models/yolov8.onnx",
    "v9": "models/yolov9.onnx",
    "v11": "models/yolov11.onnx"
}
LTIME = []

def save_txt_results(save_path, file_name, boxes, scores, class_ids):
    txt_file = os.path.join(save_path, f"{file_name}.txt")
    with open(txt_file, "w") as f:
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box
            xc = (x1 + x2) / 2
            yc = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {score:.6f}\n")
    print(f"Saved results to {txt_file}")

def process_video(model, video_path, save_path, save_video=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    os.makedirs(save_path, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if save_video:
        output_video_path = os.path.join(save_path, f"{video_name}_processed.mp4")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video FPS: {fps}, Width: {width}, Height: {height}")

        fourcc = cv2.VideoWriter_fourcc(*"XVID")  
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        if not out.isOpened():
            print("Error: Could not open video writer.")
            return

    frame_count = 0
    cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
    while cap.isOpened():
        if cv2.waitKey(1) == ord('q'):
            break

        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        boxes, scores, class_ids = model(frame)
        LTIME.append(time.time() - start)

        combined_img = draw_detections(frame, boxes, scores, class_ids)

        if save_video:
            out.write(combined_img)

        cv2.imshow("Detected Objects", combined_img)
        frame_count += 1

    cap.release()
    if save_video:
        out.release()
        print(f"Processed video saved to {output_video_path}")
    cv2.destroyAllWindows()

def process_image_or_frame(model, img_path, save_path, save_txt=False, save_img=False):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image from {img_path}")
        return
    start = time.time()
    boxes, scores, class_ids = model(img)
    LTIME.append(time.time() - start)
    combined_img = draw_detections(img, boxes, scores, class_ids)

    if save_txt:
        save_txt_results(save_path, os.path.splitext(os.path.basename(img_path))[0], boxes, scores, class_ids)

    if save_img:
        output_img_path = os.path.join(save_path, os.path.basename(img_path))
        cv2.imwrite(output_img_path, combined_img)
        print(f"Saved processed image to {output_img_path}")

    #cv2.imshow("Detected Objects", combined_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def process_directory_images(model, dir_path, save_path, save_txt=False, save_img=False):
    if not os.path.isdir(dir_path):
        print(f"Error: {dir_path} is not a valid directory.")
        return

    os.makedirs(save_path, exist_ok=True)
    for img_name in os.listdir(dir_path):
        img_path = os.path.join(dir_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping {img_name}: Unable to read image.")
            continue
        start = time.time()
        boxes, scores, class_ids = model(img)
        combined_img = draw_detections(img, boxes, scores, class_ids)
        LTIME.append(time.time() - start)
        if save_txt:
            save_txt_results(save_path, os.path.splitext(img_name)[0], boxes, scores, class_ids)

        if save_img:
            output_img_path = os.path.join(save_path, img_name)
            cv2.imwrite(output_img_path, combined_img)
            print(f"Saved processed image to {output_img_path}")

def main():
    parser = argparse.ArgumentParser(description="Process images or videos using ONNX YOLO models.")
    parser.add_argument("--type", choices=["image", "video", "images"], help="Type of input: image, video, or directory.")
    parser.add_argument("--path", type=str, help="Path to the image, video, or directory.")
    parser.add_argument("--model_type", choices=["v8", "v9", "v11"], required=True, help="YOLO model type: v8, v9, or v11.")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="Confidence threshold.")
    parser.add_argument("--iou_thres", type=float, default=0.4, help="IoU threshold.")
    parser.add_argument("--save_path", type=str, required=True, help="Directory to save results.")
    parser.add_argument("--save_txt", action="store_true", help="Save detection results in TXT format (only for images).")
    parser.add_argument("--save_img", action="store_true", help="Save images with detections drawn (only for images).")
    parser.add_argument("--save_video", action="store_true", help="Save processed video (only for video input).")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    model_path = MODEL_PATHS.get(args.model_type)
    if not model_path:
        print(f"Error: Model path for type {args.model_type} not found.")
        return

    model = YOLO(model_path, conf_thres=args.conf_thres, iou_thres=args.iou_thres)

    if args.type == "image":
        process_image_or_frame(model, args.path, args.save_path, args.save_txt, args.save_img)
    elif args.type == "video":
        process_video(model, args.path, args.save_path, save_video=args.save_video)
    elif args.type == "images":
        process_directory_images(model, args.path, args.save_path, args.save_txt, args.save_img)
    print("Number of image/s(frame) : ",len(LTIME))
    print("avg time in ms : ",(sum(LTIME)/len(LTIME))*1000)
    print("fps : ",1/(sum(LTIME)/len(LTIME)))

if __name__ == "__main__":
    main()
