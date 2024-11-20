# -*- coding: utf-8 -*-
import torch
import cv2

# load yolov5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'

# camera input 0 is laptop, 1 is iphone(external camera)
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    results = model(frame)

    
    for detection in results.xyxy[0]:  
        x1, y1, x2, y2 = map(int, detection[:4])  # gain coordinate
        conf = detection[4].item()  
        cls = int(detection[5].item())  
        label = f"{results.names[cls]} {conf:.2f}"

        # paint rectangle and distance
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # show
    cv2.imshow("YOLOv5 Detection", frame)

    # 'q' exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release camera
cap.release()
cv2.destroyAllWindows()
