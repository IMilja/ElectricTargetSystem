import cv2

while True:
    one_shot_image = cv2.imread("last_reference_image_function.jpg")
    one_shot_image = cv2.colorChange(one_shot_image, cv2.COLOR_BGR2GRAY)
    one_shot_image = cv2.threshold(one_shot_image, 0, 255,
                                   cv2.THRESH_BINARY)[1]
    _, contours, hierarchy = cv2.findContours(one_shot_image.copy(),
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        (x_position, y_position), radius = cv2.minEnclosingCircle(c)
        center = (int(x_position), int(y_position))
        if 0.5 < radius < 10 and len(approx) > 8:
            cv2.circle(one_shot_image, center, 3, [0, 0, 255], -1, cv2.LINE_AA)

    cv2.imshow('Difference', one_shot_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
