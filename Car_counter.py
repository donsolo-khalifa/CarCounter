import math
import time

import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
from sort import*



# for videos
cap = cv2.VideoCapture("cars.mp4")



model = YOLO('../yolo-weights/yolov8l.pt')

names = model.names

mask = cv2.imread('mask.png')

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [400, 297, 673, 297]

totalCount = []

while True:

    success, img = cap.read()

    # Flip the image horizontally to mirror it
    # img = cv2.flip(img, 1)

    imgRegion = cv2.bitwise_and(img,mask)

    results = model(imgRegion, stream=True)
    detections = np.empty((0, 5))

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0,0,255), 5)

    for r in results:
        boxes = r.boxes
        # class_names = boxes.cls

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            # cv2.rectangle(img, (x1,y1),(x2,y2), (255, 255, 0), 3)

            conf = box.conf[0]
            conf = math.ceil((box.conf[0] * 100)) / 100

            nameid = box.cls[0]
            class_name = names[int(nameid)]
            # print(f"Detected: {class_name}, Confidence: {conf}")

            if class_name == 'car' or class_name == 'bus' or class_name == 'motorcycle' or class_name == 'truck' and conf > 0.3 :
                # cvzone.cornerRect(img, (x1, y1, w, h), colorR=(255, 255, 0), l=15)
                # cvzone.putTextRect(img,f'{class_name} {conf}',(max(0, x1), max(35, y1)),
                #                scale=1, thickness=1, offset=5)

                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentArray))

    resultsTracker = tracker.update(detections)
    for results in resultsTracker:
        # print(f'results {results}')

        x1, y1, x2, y2, id = results
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        cvzone.cornerRect(img, (x1, y1, w, h), colorR=(255, 255, 0), l=9, rt=2)
        cvzone.putTextRect(img, f'{class_name} ', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)

        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)

        if limits[0] < cx <limits[2] and limits[1] - 15 < cy <limits[3] +15:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    cvzone.putTextRect(img, f' Vehicle count: {len(totalCount)}', (50, 50))

    cv2.imshow("Image", img)
    # cv2.imshow("Image Region", imgRegion)
    cv2.waitKey(0)


