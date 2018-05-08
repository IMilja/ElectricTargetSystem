from flask import Flask
import cv2
import numpy as np

app = Flask(__name__)


@app.route('/')
def main_page():
    return "Test"


if __name__ == '__main__':
    app.run()
