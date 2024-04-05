import os  # Importing the os module for file and directory operations

import cv2  # Importing OpenCV library
import numpy as np  # Importing NumPy library
import torch  # Importing PyTorch library
from KalmanFilter import KalmanFilter  # Importing the custom KalmanFilter class

"""
code integrates object detection, tracking, and future position prediction functionalities into a single application, 
using a combination of YOLOv5 for object detection, a custom Kalman Filter for object tracking, 
and a Dead Reckoning approach for predicting future positions of detected objects

to run the code use:  python .\kalmanDeadreckning.py --weights .\yolov9-c-converted.pt --source=0

"""

# Loading the YOLOv5 model pre-trained on the COCO dataset
model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)

def get_color_by_id(class_id):
    """
    Generates a random color based on the class ID.

    Args:
        class_id: The ID of the class.

    Returns:
        A list containing RGB values for the generated color.
    """
    np.random.seed(class_id)
    return [int(x) for x in np.random.randint(0, 255, 3)]

def run_yolov5_inference(model, frame):
    """
    Runs inference on the frame and returns detections.

    Args:
        model: The loaded YOLO model.
        frame: The current video frame.

    Returns:
        A list of detections [x1, y1, x2, y2, confidence, class_id] for each object.
    """
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

    Args:
        kf: KalmanFilter object.
        dt: Time interval.

    Returns:
        Future position (x, y).
    """
    # Extract current state
    x, y, vx, vy = kf.x.flatten()
    # Calculate future position based on current position and velocity
    future_x = x + vx * dt
    future_y = y + vy * dt
    return int(future_x), int(future_y)

def main():
    source = 0  # Video source (0 for webcam)
    cap = cv2.VideoCapture(source)  # Capturing video from the specified source

    ret, frame = cap.read()  # Reading the first frame
    if not ret:
        print("Failed to initialize video capture")
        return
    height, width = frame.shape[:2]  # Getting frame dimensions
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for video writer
    output_dir = './runs/detect/kalDeadDetection/'  # Output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Creating output directory if it doesn't exist
    out = cv2.VideoWriter(output_dir + 'output.avi', fourcc, 20.0, (width, height))  # Video writer for output

    kalman_filters = {}  # Dictionary to store Kalman filters for each class

    while cap.isOpened():
        ret, frame = cap.read()  # Reading each frame
        if not ret:
            break

        detections = run_yolov5_inference(model, frame)  # Running YOLOv5 inference to detect objects

        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls = det  # Unpacking detection values
            center_x = int((x1 + x2) / 2)  # Calculating center x-coordinate of the bounding box
            center_y = int((y1 + y2) / 2)  # Calculating center y-coordinate of the bounding box

            if cls not in kalman_filters:  # If Kalman filter for this class doesn't exist, create a new one
                kalman_filters[cls] = KalmanFilter(dt=0.1, u=0.0, std_acc=1, std_meas=0.5)

            kf = kalman_filters[cls]  # Get Kalman filter for this class
            meas = np.array([[center_x], [center_y]])  # Measurement array
            kf.update(meas)  # Update Kalman filter with measurement
            predicted = kf.predict()  # Predict future position using Kalman filter

            future_x, future_y = dead_reckoning(kf, dt=1)  # Predict future position using Dead Reckoning

            color = get_color_by_id(int(cls))  # Get color based on class ID
            cv2.circle(frame, (future_x, future_y), 10, color, -1)  # Draw circle at predicted future position
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)  # Draw bounding box

        cv2.imshow("YOLOv5 Object Tracking and Dead Reckoning", frame)  # Display frame with tracking
        out.write(frame)  # Write frame to output video

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit loop if 'q' is pressed
            break

    cap.release()  # Release video capture
    out.release()  # Release video writer
    cv2.destroyAllWindows()  # Close all OpenCV windows

if __name__ == "__main__":
    main()
