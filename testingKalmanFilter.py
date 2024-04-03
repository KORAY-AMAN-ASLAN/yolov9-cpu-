import cv2
import numpy as np
import torch
from KalmanFilter import KalmanFilter  # Ensure this class is properly implemented in KalmanFilter.py
"""
program that uses YOLOv5 for real-time object detection, integrates with a custom Kalman Filter for tracking those objects, and visualizes the results.
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


def main():
    source = 0
    cap = cv2.VideoCapture(source)  # Video source
    kalman_filters = {}  # Store Kalman Filters by class ID

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = run_yolov5_inference(model, frame)

        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls = det
            # Calculate the center of the bounding box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Initialize a Kalman Filter for new detections
            if cls not in kalman_filters:
                kalman_filters[cls] = KalmanFilter(dt=0.1, u=0.0, std_acc=1, std_meas=0.5)

            kf = kalman_filters[cls]
            meas = np.array([[center_x], [center_y]])  # Current measurement
            kf.update(meas)  # Update the Kalman Filter with the current measurement
            predicted = kf.predict()  # Predict the next position

            color = get_color_by_id(int(cls))
            # Draw the predicted position and the detected bounding box
            cv2.circle(frame, (int(predicted[0].item()), int(predicted[1].item())), 10, color, -1)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        cv2.imshow("YOLOv5 Object Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
