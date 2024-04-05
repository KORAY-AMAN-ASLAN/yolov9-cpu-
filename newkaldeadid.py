import os
import cv2
import numpy as np
import torch
from KalmanFilter import KalmanFilter

# Define the IoU function for bounding box matching
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Initialize YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)

def get_color_by_id(class_id):
    np.random.seed(class_id)
    return [int(x) for x in np.random.randint(0, 255, 3)]

def run_yolov5_inference(model, frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb)
    detections = []
    for *xyxy, conf, cls in results.xyxy[0]:
        detections.append({'bbox': [xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()], 'confidence': conf.item(), 'class_id': cls.item()})
    return detections

def dead_reckoning(kf, dt=1):
    x, y, vx, vy = kf.x.flatten()
    return int(x + vx * dt), int(y + vy * dt)

def main():
    source = 0
    cap = cv2.VideoCapture(source)
    ret, frame = cap.read()
    if not ret:
        print("Failed to initialize video capture")
        return
    output_dir = './runs/detect/kalDeadDetection/'
    os.makedirs(output_dir, exist_ok=True)
    out = cv2.VideoWriter(output_dir + 'output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame.shape[1], frame.shape[0]))

    tracked_objects = []  # To store tracked objects with their IDs and Kalman Filters
    next_id = 1  # Start IDs at 1 to ensure every object has a unique ID

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_frame_detections = run_yolov5_inference(model, frame)

        # Temporary list to store current frame's objects for ID persistence check
        current_frame_objects = []

        for det in current_frame_detections:
            x1, y1, x2, y2 = det['bbox']
            best_match = None
            best_iou_score = 0.3  # Threshold for IoU to consider a match

            # Attempt to match the current detection with existing tracked objects
            for obj in tracked_objects:
                if iou(det['bbox'], obj['bbox']) > best_iou_score:
                    best_iou_score = iou(det['bbox'], obj['bbox'])
                    best_match = obj

            if best_match:
                # Update existing tracked object
                best_match['bbox'] = det['bbox']
                best_match['kf'].update(np.array([[x1 + (x2 - x1) / 2], [y1 + (y2 - y1) / 2]]))
                current_frame_objects.append(best_match)
            else:
                # Initialize new tracked object
                kf = KalmanFilter(dt=0.1, u=0.0, std_acc=1, std_meas=0.5)
                kf.update(np.array([[x1 + (x2 - x1) / 2], [y1 + (y2 - y1) / 2]]))
                new_obj = {'id': next_id, 'bbox': det['bbox'], 'kf': kf, 'class_id': det['class_id']}
                next_id += 1
                current_frame_objects.append(new_obj)
                tracked_objects.append(new_obj)  # Add the new object to the global list of tracked objects

        # Only keep objects that were updated in this frame
        tracked_objects = current_frame_objects

        # Draw bounding boxes and IDs for each tracked object
        for obj in tracked_objects:
            bbox = obj['bbox']
            color = get_color_by_id(obj['class_id'])
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.putText(frame, f"ID: {obj['id']}", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("YOLOv5 Object Tracking and Dead Reckoning", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
