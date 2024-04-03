import cv2
import torch
import numpy as np

# Initialize YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def calculate_distance(point1, point2):
    """
    Calculate Euclidean distance between two points.
    """
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def detect_and_display(model, frame):
    """
    Perform object detection, display results, and calculate distances.
    """
    results = model(frame)

    # Extract data
    tlbr = results.xyxy[0][:, :4]  # x1, y1, x2, y2
    confidences = results.xyxy[0][:, 4]
    class_ids = results.xyxy[0][:, 5]

    for i, (box, conf, class_id) in enumerate(zip(tlbr, confidences, class_ids)):
        x1, y1, x2, y2 = map(int, box)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        # Display detections
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        cv2.putText(frame, f'{model.names[int(class_id)]} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Calculate and display distance between the first two detections
        if i == 1:
            center_x0, center_y0 = (tlbr[0][0] + tlbr[0][2]) // 2, (tlbr[0][1] + tlbr[0][3]) // 2
            distance = calculate_distance((center_x, center_y), (center_x0, center_y0))
            cv2.putText(frame, f'Distance: {distance:.2f}px', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow('YOLOv5 Detection', frame)

def main():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect and display
        detect_and_display(model, frame_rgb)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
