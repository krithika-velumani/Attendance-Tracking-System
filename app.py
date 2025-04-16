import face_recognition
import cv2
import csv
from datetime import datetime
import os
import time
import urllib.request
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit

# Initialize Flask and SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# Load known face encodings
image_1 = face_recognition.load_image_file("1.jpg")
image_1_face_encoding = face_recognition.face_encodings(image_1)[0]
image_2 = face_recognition.load_image_file("2.jpg")
image_2_face_encoding = face_recognition.face_encodings(image_2)[0]
image_3 = face_recognition.load_image_file("3.jpg")
image_3_face_encoding = face_recognition.face_encodings(image_3)[0]


# Known faces
known_face_encodings = [
    image_1_face_encoding,
    image_2_face_encoding,
    image_3_face_encoding,
]
known_face_names = ["ARUL", "CHARLIE", "ALEX"]

# Create a directory for images if it doesn't exist
os.makedirs('static', exist_ok=True)

# Video capture from webcam
cap = cv2.VideoCapture(0)

# Function to write data to CSV
def write_to_csv(data):
    file_exists = os.path.isfile("attendance_data.csv")
    with open("attendance_data.csv", mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["name", "date", "time"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

# Function to generate frames for video feed
def gen():
    detected_faces = set()
    start_time = time.time()
    duration = 5  # Duration for face check in seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                detected_faces.add(name)
            face_names.append(name)

        # Drawing face locations and names
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Check for missed faces and send to ThingSpeak
        if time.time() - start_time >= duration:
            missed_faces = list(set(known_face_names) - detected_faces)
            if missed_faces:
                now = datetime.now()
                for missed_face in missed_faces:
                    data = {"name": missed_face, "date": str(now.date()), "time": str(now.time())}
                    write_to_csv(data)

                missed_faces_str = ",".join(missed_faces)
                api_key = "R2NW4B1FJ36QCRH2"
                url = f"https://api.thingspeak.com/update?api_key={api_key}&field1={missed_faces_str}"
                conn = urllib.request.urlopen(url)
                conn.close()

        # Send video frame to the frontend
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            break
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
