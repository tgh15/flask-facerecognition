from flask import Flask, jsonify, request, render_template, redirect, session
from flask_session import Session
import cv2 as cv
import numpy as np
import base64
import os
from datetime import datetime

from simple_facerec import SimpleFacerec

app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

img_path = os.path.join(os.path.dirname(__file__), 'train')


def b64decode(data):
    imgbase64 = data.replace("data:image/jpeg;base64,", "")
    decoded = base64.b64decode(imgbase64)
    np_data = np.fromstring(decoded, np.uint8)
    return cv.imdecode(np_data, cv.IMREAD_UNCHANGED)


def b64encode(img):
    retval, buffer = cv.imencode('.jpg', img)
    base64_bytes = base64.b64encode(buffer)
    return base64_bytes.decode('utf-8')


@app.route('/', methods=['GET'])
def index():
    if not session.get("name"):
        return redirect("/login")
    return render_template('index.html', title="Dashboard", username=session.get('name'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get("name"):
        return redirect("/")

    if request.method == 'POST':
        content = request.get_json()
        imgb64 = content['image']
        image = b64decode(imgb64)

        face_locations, face_names = sfr.detect_known_faces(image)
        if len(face_names) != 1:
            return jsonify({'success': False, 'message': 'Detection Failed! Please Try Again!', 'data': face_names})
        if face_names[0] == 'Unknown':
            return jsonify({'success': False, 'message': 'You are not registered', 'data': face_names})
        session["name"] = face_names[0]
        return jsonify({'success': True, 'message': 'Login Successfully', 'data': face_names})
    return render_template('login.html', title="Login")


@app.route('/register', methods=['GET', 'POST'])
def register():
    if session.get("name"):
        return redirect("/")

    if request.method == 'POST':
        content = request.get_json()
        imgb64 = content['image']
        username = content['username']

        image = b64decode(imgb64)

        if os.path.exists(os.path.join(img_path, username+'.jpg')):
            return jsonify({'success': False, 'message': 'User already registered'})

        if not os.path.exists(img_path):
            os.makedirs(img_path)
        cv.imwrite(os.path.join(img_path, username+'.jpg'), image)
        sfr.load_encoding_image('train/'+username+'.jpg')
        return jsonify({'success': True, 'message': 'Register Successfully'})

    return render_template('register.html', title="Register")


@app.route("/presensi", methods=['GET', 'POST'])
def presensi():
    if request.method == 'POST':
        content = request.get_json()
        imgb64 = content['image']
        image = b64decode(imgb64)

        face_locations, face_names = sfr.detect_known_faces(image)
        now = datetime.now()
        sesi = datetime.strptime("06:30:00", "%H:%M:%S")
        current_time = now.strftime("%H:%M:%S")
        enter = datetime.strptime(current_time, "%H:%M:%S")
        if (enter > sesi):
            ket = "Terlambat"
        else:
            ket = "Hadir"
        if len(face_names) != 1:
            return jsonify({'success': False, 'message': 'Detection Failed! Please Try Again!', 'data': face_names})

        if face_names[0] == 'Unknown':
            return jsonify({'success': False, 'message': 'Detection Failed! Please Try Again!', 'data': face_names})

        return jsonify({'success': True, 'message': 'Presensi Sukses', 'data': {'face_names': face_names, 'attendance_time': current_time, 'keterangan': ket}})

    return render_template('presensi.html', title="Presensi")


@app.route("/logout")
def logout():
    session["name"] = None
    return redirect("/")


@app.route('/video', methods=['POST'])
def video():
    content = request.get_json()
    data = content['image']

    img = b64decode(data)

    face = cv.CascadeClassifier('haarcascade/face.xml')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces_rect = face.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6)
    for (x, y, w, h) in faces_rect:
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 60), thickness=2)

    img64 = b64encode(img)

    return jsonify({'image': img64, 'face_detected': len(faces_rect)})


if __name__ == '__main__':
    sfr = SimpleFacerec()
    sfr.load_encoding_images("train/")
    app.run(debug=True)
