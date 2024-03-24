from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'Error': 'No file part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'Error': 'No file selected'}), 400

    img_stream = file.read()
    nparr = np.fromstring(img_stream, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Save the uploaded image to a temporary directory
    temp_dir = os.path.join(app.root_path, 'static', 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    res_filename = secure_filename(file.filename)
    result_filepath = os.path.join(temp_dir, res_filename)
    cv2.imwrite(result_filepath, img)

    # Generate the URL for the result image
    result_image_url = url_for('static', filename='temp/' + res_filename)

    return render_template("result.html",filename=res_filename, result_image_url=result_image_url)

if __name__ == '__main__':
    app.run(debug=True)
