import cv2
import numpy as np


def detect_shots(transformed_img, active_reference_image):
    canny_img = cv2.Canny(transformed_img, 75, 150, True)
    _, contours, hierarchy = cv2.findContours(canny_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        (x_position, y_position), radius = cv2.minEnclosingCircle(c)
        center = (int(x_position), int(y_position))
        cv2.circle(active_reference_image, center, radius, [255, 255, 255], -1)

    return active_reference_image


def find_last_shot(active_reference_image, last_reference_image):
    image_difference = find_image_difference(active_reference_image, last_reference_image)
    rotated_image = cv2.flip(image_difference, 1)
    canny_image = cv2.Canny(rotated_image, 75, 150)
    _, contours, hierarchy = cv2.findContours(canny_image.copy(),
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)
    return _, contours, hierarchy


def find_image_difference(active_reference_image, last_reference_image):
    last_reference_image = cv2.threshold(last_reference_image, 127, 255,
                                         cv2.THRESH_BINARY_INV)[1]
    active_reference_image = cv2.threshold(active_reference_image,
                                           127, 255,
                                           cv2.THRESH_BINARY)[1]
    image_difference = cv2.bitwise_and(active_reference_image, last_reference_image)

    return image_difference
