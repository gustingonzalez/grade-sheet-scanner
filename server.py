import subprocess
import cv2
from flask import Flask, render_template, jsonify
from flask_cors import CORS

import image.processor as image_processor
import digit_predictor

from settings import (
    SCAN_IMAGE_MOCK,
    SCANNED_IMAGE_PATH,
    SCAN_COMMAND_ADF,
    SCAN_COMMAND,
    SCAN_TIMEOUT
)

app = Flask(__name__, template_folder="www/templates",
            static_folder='www/static')

# Allows all origins.
CORS(app, origins='*')


def scan_image():
    scan_cmd = SCAN_COMMAND_ADF

    p = subprocess.Popen(f"{scan_cmd}", shell=True)
    try:
        p.wait(SCAN_TIMEOUT)
    except subprocess.TimeoutExpired:
        p.kill()
        raise Exception('Scan timeout reached.')

    img = cv2.imread(SCANNED_IMAGE_PATH)
    return img


@app.route("/scan_and_process_image")
def scan_and_process_image():
    if SCAN_IMAGE_MOCK is not None:
        print("INFO: Working with an image mock...")
        image = cv2.imread(SCAN_IMAGE_MOCK)
    else:
        print("INFO: Working with a scanner...")
        image = scan_image()
    digits_images = image_processor.extract_digits_images(image)

    grades_by_student = []
    for i in range(0, len(digits_images), 2):
        digit1_image = digits_images[i]
        digit2_image = digits_images[i + 1]

        digit1 = ""
        digit2 = ""
        if digit1_image is not None:
            # Removes leading '0', if predicted.
            if (digit1 := digit_predictor.predict(digit1_image)) == "0":
                digit1 = ""

        if digit2_image is not None:
            digit2 = digit_predictor.predict(digit2_image)

        digit = digit1 + digit2
        grades_by_student.append(digit)
    return jsonify(grades_by_student)


@app.route('/')
def index():
    return render_template('index.html')


def main():
    print("Starting server...")
    app.run()


if __name__ == "__main__":
    main()
