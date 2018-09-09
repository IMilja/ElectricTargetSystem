import cv2
import numpy as np


def detect_shots(transformed_img, active_reference_image):
    canny_img = cv2.Canny(transformed_img.copy(), 50, 150, True)
    cv2.imwrite('Threshold_image.jpg', canny_img)
    target_empty = cv2.imread('last_reference_image_circle.png')
    target_empty = cv2.bitwise_not(target_empty)
    _, contours, hierarchy = cv2.findContours(canny_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        (x_position, y_position), radius = cv2.minEnclosingCircle(c)
        center = (int(x_position), int(y_position))
        if 0.5 < radius < 10 and len(approx) > 8:
            cv2.circle(active_reference_image, center, 3, [255, 255, 255], -1, cv2.LINE_AA)
            cv2.drawMarker(target_empty, center, [0, 0, 255], cv2.MARKER_CROSS, 10, 2)
    target_empty = cv2.flip(target_empty, 1)
    cv2.imwrite("static/Pictures/Target_Mete_Puna.png", target_empty)

    return active_reference_image


def find_last_shot(active_reference_image, last_reference_image):
    image_difference = find_image_difference(active_reference_image, last_reference_image)
    rotated_image = cv2.flip(image_difference, 1)
    _, contours, hierarchy = cv2.findContours(rotated_image.copy(),
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)
    return _, contours, hierarchy


def find_image_difference(active_reference_image, last_reference_image):
    last_reference_image = cv2.threshold(last_reference_image, 0, 255,
                                         cv2.THRESH_BINARY)[1]
    cv2.imwrite("last_reference_image_function.jpg", last_reference_image)
    active_reference_image = cv2.threshold(active_reference_image,
                                           0, 255,
                                           cv2.THRESH_BINARY)[1]
    cv2.imwrite("active_reference_image_function.jpg", active_reference_image)
    image_difference = cv2.bitwise_xor(active_reference_image, last_reference_image)
    kernel = np.ones((4, 4), np.uint8)
    image_difference = cv2.morphologyEx(image_difference, cv2.MORPH_OPEN, kernel)
    cv2.imwrite("image_difference.jpg", image_difference)
    return image_difference


def score_shot(x_position, y_position, width, height, target, target_size):
    x_position = int(x_position)
    y_position = int(y_position)

    print("Y_POZICIJA:" + str(y_position))
    print("X_POZICIJA:" + str(x_position))

    x_position = (-(width / 2) + x_position)
    y_position = ((height / 2) - y_position)

    print("Y_POZICIJA:" + str(y_position))
    print("X_POZICIJA:" + str(x_position))

    distance_from_center = np.sqrt(x_position ** 2 + y_position ** 2)
    distance_from_center = int(distance_from_center)
    print(distance_from_center)
    print(target)
    if distance_from_center < target[0]:
        score = 10
    elif distance_from_center < target[1]:
        score = 9
    elif distance_from_center < target[2]:
        score = 8
    elif distance_from_center < target[3]:
        score = 7
    elif distance_from_center < target[4]:
        score = 6
    elif distance_from_center < target[5]:
        score = 5
    elif distance_from_center < target[6]:
        score = 4
    elif distance_from_center < target[7]:
        score = 3
    elif distance_from_center < target[8]:
        score = 2
    elif distance_from_center < target[9]:
        score = 1
    else:
        score = 0
    return score, round(x_position*(target_size / width), 2), round(y_position*(target_size / height), 2)
