import cv2
import numpy as np
import math
def nothing(x):
    pass

# Camera port declaration and ini
cameraPort = 1
camera = cv2.VideoCapture(cameraPort)
cv2.namedWindow('Edges')

cv2.createTrackbar('lower', 'Edges', 1, 255, nothing)
cv2.createTrackbar('higher', 'Edges', 1, 255, nothing)

while camera.isOpened():
    lower = cv2.getTrackbarPos('lower', 'Edges')
    higher = cv2.getTrackbarPos('higher', 'Edges')
    ret, picture = camera.read()
    grayScale = cv2.cvtColor(picture.copy(), cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(grayScale, 5, 100, 100)
    edges = cv2.Canny(blurred, lower, higher)
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLines(edges, 1, np.pi / 180.0, 150, None, 0, 0)
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0, y0 = a * rho, b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
            cv2.line(picture, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Camera', picture)
    cv2.imshow('Gray Scale', grayScale)
    cv2.imshow('Edges', edges)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
