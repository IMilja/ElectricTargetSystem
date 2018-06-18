from flask import Flask, render_template, send_from_directory, redirect
import cv2
from Calibration import *
from Detection import *

app = Flask(__name__)

# global parameters for image correction and shot detection
allShots = []
reset = True
source_pts = []
dst_points = []
cnt = None

# declare the camera port and init our camera
camera_port = 0
camera = cv2.VideoCapture(camera_port)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/resetState')
def reset_state():
    global allShots
    global reset
    del allShots[:]
    reset = True
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
    reference_image = None
    ret, source_image = camera.read()
    gray_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    corrected_image = image_perspective_correction(gray_image, source_pts, dst_points, cnt)
    reference_image = detect_shots(corrected_image, reference_image)




@app.route('/settings')
def settings():
    render_template('Settings.html')


@app.route('/getCalibrationParams')
def get_calibration_params():
    ret, source_image = camera.read()
    global source_pts
    global dst_points
    global cnt
    source_pts, dst_points, cnt = get_correction_parameters(source_image)


if __name__ == '__main__':
    app.run()
