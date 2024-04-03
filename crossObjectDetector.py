import cv2
import numpy as np
import torch
import winsound  # For Windows only, for cross-platform use an alternative

# Assuming KalmanFilter class is defined in kalman_filter.py
from KalmanFilter import KalmanFilter

# Load the YOLOv5 model pre-trained on COCO dataset
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


def get_color_by_id(class_id):
    """
    Generates a unique color for each class ID to ensure consistency across runs.
    """
    np.random.seed(class_id)
    return [int(x) for x in np.random.randint(0, 255, 3)]

#python   CrossFilterDetector.py  --source 0

def run_yolov5_inference(model, frame):
    """
    Performs YOLOv5 inference on a frame and returns detections.
    Each detection includes bounding box coordinates, confidence score, and class ID.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb)
    detections = []
    for *xyxy, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = xyxy
        detections.append([x1.item(), y1.item(), x2.item(), y2.item(), conf.item(), cls.item()])
    return detections


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
            x1, y1, _, _, _, _ = det1
            x2, y2, _, _, _, _ = det2
            distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if distance < threshold:
                return True
    return False


def main():
    source = 0
    cap = cv2.VideoCapture(source)
    kalman_filters = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = run_yolov5_inference(model, frame)

        # Check for proximity and alert if conditions are met
        if check_proximity(detections):
            beep_alert()  # Plays a beep sound if any two objects are too close

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

            color = get_color_by_id(int(cls))
            cv2.circle(frame, (int(predicted[0].item()), int(predicted[1].item())), 10, color, -1)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        cv2.imshow("YOLOv5 Object Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
# python   crossObjectDetector.py  --source 0