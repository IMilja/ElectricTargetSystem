import cv2
import numpy as np
import math

# Sorts our points so the picture is correctly rotated


def order_points(source_points):
    dst_points = np.zeros((4, 2), dtype="float32")
    s = source_points.sum(axis=1)

    dst_points[0] = source_points[np.argmin(s)]
    dst_points[2] = source_points[np.argmax(s)]
    difference = np.diff(source_points, axis=1)
    dst_points[1] = source_points[np.argmin(difference)]
    dst_points[3] = source_points[np.argmax(difference)]

    return dst_points


def get_correction_parameters(source_img):
    x_coordinate = 0
    y_coordinate = 1
    source_img = cv2.medianBlur(source_img, 3)
    gray_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    ret, thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # check if contours exist
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        x, y, width, height = cv2.boundingRect(cnt)
        if len(approx) == 4:
            source_pts = [(approx[0][0][x_coordinate], approx[0][0][y_coordinate]),
                          (approx[1][0][x_coordinate], approx[1][0][y_coordinate]),
                          (approx[2][0][x_coordinate], approx[2][0][y_coordinate]),
                          (approx[3][0][x_coordinate], approx[3][0][y_coordinate])]

            # order our points by converting them first from a normal list into a numpy array
            source_pts = np.asarray(source_pts, np.float32)
            source_pts = order_points(source_pts)

            # create a new numpy array of our desired image format
            dst_points = np.array([
                [0, 0],
                [x + width - 1, 0],
                [x + width - 1, y + height - 1],
                [0, y + height - 1]], dtype="float32")

            return source_pts, dst_points, cnt


def image_perspective_correction(gray_img, source_pts, dst_pts, cnt):
    x, y, width, height = cv2.boundingRect(cnt)
    transform_matrix = cv2.getPerspectiveTransform(source_pts, dst_pts)
    transformed_img = cv2.warpPerspective(gray_img, transform_matrix, (x + width - 1, y + height - 1), gray_img.size)

    return transformed_img




