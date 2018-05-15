import cv2
import numpy as np

# Camera port declaration and ini
cameraPort = 1
camera = cv2.VideoCapture(cameraPort)

while camera.isOpened():
    ret, picture = camera.read()
    grayScale = cv2.cvtColor(picture.copy(), cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(grayScale, 5, 100, 100)
    edges = cv2.Canny(blurred, 125, 150)
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    if lines is not None:
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(picture, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow('Camera', picture)
    cv2.imshow('Gray Scale', grayScale)
    cv2.imshow('Edges', edges)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
