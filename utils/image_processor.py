import cv2
import numpy as np

def load_image(video_path):
    cap = cv2.VideoCapture(video_path)
    images = []
    while True:
        ret, img = cap.read()
        if not ret:
            break
        images.append(img)
    
    return np.array(images)
