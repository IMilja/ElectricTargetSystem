import cv2
import numpy as np
import math


def nothing(x):
    pass


def acceptLinePair(line1, line2, minTheta):
    thetaLine1 = line1[0][1]
    thetaLine2 = line2[0][1]

    if thetaLine1 < minTheta:
        thetaLine1 += np.pi

    if thetaLine2 < minTheta:
        thetaLine2 += np.pi

    return abs(thetaLine1 - thetaLine2) > minTheta


def computeIntersect(line1, line2):
    p1 = lineToPointPair(line1)
    p2 = lineToPointPair(line2)

    denom = (p1[0][0] - p1[1][0]) * (p2[0][1] - p2[1][1]) - (p1[0][1] - p1[1][1]) * (p2[0][0] - p2[1][0])
    intersect = (((p1[0][0]*p1[1][1] - p1[0][1]*p1[1][0])*(p2[0][0] - p2[1][0]) -
                  (p1[0][0] - p1[1][0])*(p2[0][0]*p2[1][1] - p2[0][1]*p2[1][0])) / denom,
                 ((p1[0][0]*p1[1][1] - p1[0][1]*p1[1][0])*(p2[0][1] - p2[1][1]) -
                  (p1[0][1] - p1[1][1])*(p2[0][0]*p2[1][1] - p2[0][1]*p2[1][0])) / denom)

    return intersect


def lineToPointPair(line):
    points = []
    rho = line[0][0]
    theta = line[0][1]

    cosTheta = math.cos(theta)
    sinTheta = math.sin(theta)

    x0 = rho*cosTheta
    y0 = rho*sinTheta
    alpha = 1000

    points.append((x0 + alpha * (-sinTheta), int(y0 + alpha * cosTheta)))
    points.append((x0 - alpha * (-sinTheta), int(y0 - alpha * cosTheta)))

    return points


# Camera port declaration and ini
cameraPort = 1
camera = cv2.VideoCapture(cameraPort)
cv2.namedWindow('Edges')
cv2.createTrackbar('houghParms', 'Edges', 150, 255, nothing)

while camera.isOpened():
    houghParms = cv2.getTrackbarPos('houghParms', 'Edges')
    ret, picture = camera.read()
    grayScale = cv2.cvtColor(picture.copy(), cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(grayScale, 7, 100, 100)
    edges = cv2.Canny(blurred, 70, 100)
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLines(edges, 1, np.pi / 180.0, houghParms, None, 0, 0)
    intersections = []
    print("LEN: " + str(len(lines)))
    if lines is not None:
        for i in range(0, len(lines)):
            for j in range(0, len(lines)):
                line1 = lines[i]
                line2 = lines[j]
                if acceptLinePair(line1, line2, np.pi / 32):
                    intersection = computeIntersect(line1, line2)
                    intersections.append(intersection)

    if intersections is not None:
        for intersect in intersections:
            center = (int(intersect[0]), int(intersect[1]))
            cv2.circle(picture, center, 5, (0, 255, 0), -1)

    cv2.imshow('Camera', picture)
    cv2.imshow('Gray Scale', grayScale)
    cv2.imshow('Edges', edges)
    if cv2.waitKey(1) & 0xFF == ord('p'):
        print(intersections)
        intersections.sort()
        print(intersections)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
