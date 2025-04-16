import face_recognition
import cv2
import csv
from datetime import datetime
import os
import time
import urllib.request

# Replace with your webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

# Load known face encodings
image_1 = face_recognition.load_image_file("1.jpg")
image_1_face_encoding = face_recognition.face_encodings(image_1)[0]
image_2 = face_recognition.load_image_file("2.jpg")
image_2_face_encoding = face_recognition.face_encodings(image_2)[0]
image_3 = face_recognition.load_image_file("3.jpg")
image_3_face_encoding = face_recognition.face_encodings(image_3)[0]

# List of known face encodings and names
known_face_encodings = [
    image_1_face_encoding,
    image_2_face_encoding,
    image_3_face_encoding,
]
known_face_names = ["ARUL", "CHARLIE", "ALEX"]

# Function to write data to CSV
def write_to_csv(data):
    file_exists = os.path.isfile("attendance_data.csv")
    
    with open("attendance_data.csv", mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["name", "date", "time"])
        
        # Write header if the file doesn't exist
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(data)

# Track faces detected
detected_faces = set()
start_time = time.time()  # Record the start time
duration = 5  # 2 minutes in seconds
checked_once = False  # Flag to check faces only once

# Main loop
while True:
    try:
        # Fetch frame from webcam
        ret, frame = cap.read()  # Read the frame from webcam
        if not ret:
            print("Failed to grab frame")
            break

        # Resize the frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Process the frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            
            # If a match is found, use the first match's name
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                detected_faces.add(name)  # Mark this face as detected
                
            face_names.append(name)

        # Display face locations and names on the video feed
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Show the video feed
        cv2.imshow('Webcam', frame)
        missed_faces = []  # List to store missed faces

        # Check if 2 minutes have passed and check faces only once
        if time.time() - start_time >= duration and not checked_once:
            # Faces that were not detected in the last 2 minutes
            missed_faces = list(set(known_face_names) - detected_faces)
            if missed_faces:
                # Log the missed faces to the CSV
                now = datetime.now()
                for missed_face in missed_faces:
                    data = {
                        "name": missed_face,
                        "date": str(now.date()),
                        "time": str(now.time())
                    }
                    write_to_csv(data)
                    print(f"Missed face: {missed_face}")

                # Send missed faces to ThingSpeak
                missed_faces_str = ",".join(missed_faces)  # Join missed faces as a string
                api_key = "R2NW4B1FJ36QCRH2"  # Replace with your ThingSpeak API key
                url = f"https://api.thingspeak.com/update?api_key={api_key}&field1={missed_faces_str}"
                
                # Send HTTP request
                conn = urllib.request.urlopen(url)
                print(f"HTTP status code={conn.getcode()}")
                conn.close()
            
            # Set the flag to prevent repeated checking
            checked_once = True

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(f"Error: {e}")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
