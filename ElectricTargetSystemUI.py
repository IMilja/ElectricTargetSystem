from flask import Flask, render_template, redirect, request
import cv2
from Calibration import *
from Detection import *

app = Flask(__name__)

# global parameters for image correction and shot detection
all_shots = []
reset = True
source_pts = []
dst_points = []
cnt = None
target = []
target_size = 0
field_size = 0

# make a reference image to save our shots
last_reference_image = None
# declare the camera port and init our camera
camera_port = 0
camera = cv2.VideoCapture(camera_port)


@app.route('/')
def index():
    print(target)
    return render_template('index.html')


@app.route('/resetState')
def reset_state():
    global all_shots
    global reset
    global last_reference_image
    del all_shots[:]
    reset = True
    last_reference_image = np.zeros((int(dst_points[2][1]), int(dst_points[2][0]), 1), np.uint8)
    return redirect("/primaryState")


@app.route('/primaryState')
def primary_state():
    return render_template('TargetView.html',
                           shotCount=0,
                           shots=0,
                           result=0)


@app.route('/aboutUs')
def about_us():
    return render_template('AboutUs.html')


@app.route('/searchShot')
def search_shot():
    global last_reference_image
    global source_pts
    global dst_points
    global cnt
    global target
    ret, source_image = camera.read()
    gray_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    reference_image = np.zeros((int(dst_points[2][1]), int(dst_points[2][0]), 1), np.uint8)
    corrected_image = image_perspective_correction(gray_image, source_pts, dst_points, cnt)
    reference_image = detect_shots(corrected_image, reference_image)
    last_shot = find_last_shot(reference_image, last_reference_image)[1]
    last_reference_image = reference_image.copy()
    if last_shot:
        all_shots.append(last_shot[0])

    shots = []
    result = 0
    shot_count = 0

    for shot in all_shots:
        shot_count += 1
        (x_position, y_position), radius = cv2.minEnclosingCircle(shot)
        score, x_position, y_position = score_shot(x_position,
                                                   y_position,
                                                   int(dst_points[2][0]),
                                                   int(dst_points[2][1]),
                                                   target, target_size)
        shots.append((score, x_position, y_position))
        result += int(score)

    return render_template('TargetView.html', shotCount=shot_count, shots=shots, result=result)


@app.route('/settings')
def settings():
    return render_template('Settings.html', target_size=target_size, field_size=field_size)


@app.route('/getCalibrationParams',  methods=['GET', 'POST'])
def get_calibration_params():
    global source_pts
    global dst_points
    global cnt
    global width
    global height
    global x
    global y
    global last_reference_image
    global target
    global target_size
    global field_size
    ret, source_image = camera.read()
    source_pts, dst_points, cnt = get_correction_parameters(source_image)
    x, y, width, height = cv2.boundingRect(cnt)
    last_reference_image = np.zeros((int(dst_points[2][1]), int(dst_points[2][0]), 1), np.uint8)

    target = []
    target_size = float(request.form['target_size'])
    field_size = float(request.form['field_size'])
    target_width_pixels = dst_points[2][0] / target_size
    target_height_pixels = dst_points[2][1] / target_size
    target_pixels = np.sqrt(target_width_pixels ** 2 + target_height_pixels ** 2)
    start_point = 0
    for i in range(10):
        start_point += target_pixels * field_size
        target.append(round(start_point, 2))
    print(target)

    return redirect('/settings')


if __name__ == '__main__':
    app.run()
