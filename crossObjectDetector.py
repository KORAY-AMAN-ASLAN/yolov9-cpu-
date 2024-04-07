import cv2
import numpy as np
import torch
import winsound
from ultralytics import YOLO

from KalmanFilter import KalmanFilter
''
"""
This Python script integrates YOLOv5 for object detection, Kalman filtering for object tracking,
 dead reckoning for predicting future positions, and audio alerts for detecting close object proximity.
 , this code can monitor the movements, detect any anomalies or collisions, and trigger alerts for maintenance or safety measures.
To run use this:  python .\crossObjectDetector.py --weights .\yolov9-c-converted.pt --source=0   


"""
# Load the YOLOv5 model pre-trained on COCO dataset
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

#model = YOLO("yolov8n.pt")
def get_color_by_id(class_id):
    """
    Generates a unique color for each class ID to ensure consistency across runs.
    """
    np.random.seed(class_id)
    return [int(x) for x in np.random.randint(0, 255, 3)]


def run_yolov5_inference(model, frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print(frame_rgb)
    results = model(frame_rgb)
    detections = []
    for *xyxy, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = xyxy
        class_name = model.names[int(cls.item())]  # Get class name
        detections.append([x1.item(), y1.item(), x2.item(), y2.item(), conf.item(), cls.item(), class_name])
    return detections


def dead_reckoning(kf, dt=1):
    """
    Predicts future position based on current velocity using Dead Reckoning.
    Assumes the state vector format is [x, y, vx, vy].T.
    """
    x, y, vx, vy = kf.x.flatten()
    future_x = x + vx * dt
    future_y = y + vy * dt
    return int(future_x), int(future_y)

def beep_alert():
    """
    Plays a beep sound as an alert. Adjust frequency and duration as needed.
    """
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 1000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)

def check_proximity(detections, threshold=50):
    """
    Checks if any two detections are within a specified proximity threshold.
    Returns True if any two detections are close to each other.
    """
    for i, det1 in enumerate(detections):
        for j, det2 in enumerate(detections):
            if i >= j:  # Avoid repeating comparisons
                continue
            x1, y1, _, _, _, _, _ = det1  # Update to match the new detections structure
            x2, y2, _, _, _, _, _ = det2  # Update to match the new detections structure
            distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if distance < threshold:
                return True
    return False


def main():
    source = 0  # Video capture source
    cap = cv2.VideoCapture(source)
    kalman_filters = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = run_yolov5_inference(model, frame)

        # Update Kalman filters and predict future positions
        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls, class_name = det  # Extract class_name here

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
            # Display class name and probability
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


        if check_proximity(detections):
            beep_alert()  # Generate audio alert if objects are too close

        cv2.imshow("Cross Object Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
