import cv2
import numpy as np
import torch
from KalmanFilter import KalmanFilter  # Ensure this class includes velocity in its state

"""
code integrates object detection, tracking, and future position prediction functionalities into a single application, 
using a combination of YOLOv5 for object detection, a custom Kalman Filter for object tracking, 
and a Dead Reckoning approach for predicting future positions of detected objects

"""
# Load the YOLOv5 model pre-trained on the COCO dataset
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def get_color_by_id(class_id):
    np.random.seed(class_id)
    return [int(x) for x in np.random.randint(0, 255, 3)]

def run_yolov5_inference(model, frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
    results = model(frame_rgb)  # Run inference
    detections = []  # Store detections
    # Extract bounding box coordinates, confidence score, and class ID for each detection
    for *xyxy, conf, cls in results.xyxy[0]:
        detections.append([xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item(), conf.item(), cls.item()])
    return detections


def dead_reckoning(kf, dt=1):
    """
    Predicts future position based on current velocity using Dead Reckoning.
    Assumes the state vector format is [x, y, vx, vy].T
    """
    # Extract current state
    x, y, vx, vy = kf.x.flatten()
    # Calculate future position based on current position and velocity
    future_x = x + vx * dt
    future_y = y + vy * dt
    return int(future_x), int(future_y)

def main():
    source = 0
    cap = cv2.VideoCapture(source)
    kalman_filters = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        detections = run_yolov5_inference(model, frame)

        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls = det
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            if cls not in kalman_filters:
                kalman_filters[cls] = KalmanFilter(dt=0.1, u=0.0, std_acc=1, std_meas=0.5)

            kf = kalman_filters[cls]

            meas = np.array([[center_x], [center_y]])
            kf.update(meas)
            predicted = kf.predict()

            future_x, future_y = dead_reckoning(kf, dt=1)

            color = get_color_by_id(int(cls))
            cv2.circle(frame, (future_x, future_y), 10, color, -1)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Display the frame
        cv2.imshow("YOLOv5 Object Tracking and Dead Reckoning", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
