import cv2
import numpy as np
import os
import os.path
import time

IMAGE_DIR = 'Aberdeen'
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

total_time = 0
n_runs = 0

def extract_roi(img, rect):
    x, y, w, h = rect
    return img[y:y+h, x:x+w]


def process_image(image):
    start = time.time()
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey_image, 1.3, 5)
    for rect in faces:
        fx, fy, fw, fh = rect
        cv2.rectangle(image, (fx, fy), (fx+fw, fy+fh), (255,0,0), 2)
        roi_grey = extract_roi(grey_image, rect)
        roi_color = extract_roi(image, rect)
        eyes = eye_cascade.detectMultiScale(roi_grey)
        for ex, ey, ew, eh in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)
    global total_time, n_runs
    end = time.time()
    total_time += end - start
    n_runs += 1
    # print(total_time / n_runs)
    cv2.imshow('img', image)

# image_files = (os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR))



# for image_file in image_files:
#     image = cv2.imread(image_file)
#     process_image(image)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    process_image(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

