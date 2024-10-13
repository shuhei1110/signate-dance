import cv2
import numpy as np

def load_image(video_path):
    cap = cv2.VideoCapture(video_path)
    images = []

    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    images.append(img[y:y+h, x:x+w])

    while True:
        ret, img = cap.read()

        if not ret:
            break
        images.append(img[y:y+h, x:x+w])
    
    return np.array(images)


def save_video(images, output_path, fps=30):
    h, w, c = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for image in images:
        video.write(image)

    video.release()