from src.face_detector import YOLOv5
from src.FaceAntiSpoofing import AntiSpoof 
import cv2
import numpy as np
import argparse


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
image_2 = face_recognition.load_image_file("2.jpg")
image_2_face_encoding = face_recognition.face_encodings(image_2)[0]
image_3 = face_recognition.load_image_file("3.jpg")
image_3_face_encoding = face_recognition.face_encodings(image_3)[0]
image_4 = face_recognition.load_image_file("4.jpg")
image_4_face_encoding = face_recognition.face_encodings(image_4)[0]

# List of known face encodings and names
known_face_encodings = [
    image_2_face_encoding,
    image_3_face_encoding,
    image_4_face_encoding,
]
known_face_names = ["PRIYA", "KRITHIKA", "LAKSHANA"]


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


COLOR_REAL = (0, 255, 0)
COLOR_FAKE = (0, 0, 255)
COLOR_UNKNOWN = (127, 127, 127)

def increased_crop(img, bbox : tuple, bbox_inc : float = 1.5):
    # Crop face based on its bounding box
    real_h, real_w = img.shape[:2]
    
    x, y, w, h = bbox
    w, h = w - x, h - y
    l = max(w, h)
    
    xc, yc = x + w/2, y + h/2
    x, y = int(xc - l*bbox_inc/2), int(yc - l*bbox_inc/2)
    x1 = 0 if x < 0 else x 
    y1 = 0 if y < 0 else y
    x2 = real_w if x + l*bbox_inc > real_w else x + int(l*bbox_inc)
    y2 = real_h if y + l*bbox_inc > real_h else y + int(l*bbox_inc)
    
    img = img[y1:y2,x1:x2,:]
    img = cv2.copyMakeBorder(img, 
                             y1-y, int(l*bbox_inc-y2+y), 
                             x1-x, int(l*bbox_inc)-x2+x, 
                             cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img

def make_prediction(img, face_detector, anti_spoof):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 
    bbox = face_detector([img])[0]
    
    if bbox.shape[0] > 0:
        bbox = bbox.flatten()[:4].astype(int)
    else:
        return None

    pred = anti_spoof([increased_crop(img, bbox, bbox_inc=1.5)])[0]
    score = pred[0][0]
    label = np.argmax(pred)   
    
    return bbox, label, score

if __name__ == "__main__":
    # parsing arguments
    def check_zero_to_one(value):
        fvalue = float(value)
        if fvalue <= 0 or fvalue >= 1:
            raise argparse.ArgumentTypeError("%s is an invalid value" % value)
        return fvalue
    
    p = argparse.ArgumentParser(
        description="Spoofing attack detection on videostream")
    p.add_argument("--input", "-i", type=str, default=None, 
                   help="Path to video for predictions")
    p.add_argument("--output", "-o", type=str, default=None,
                   help="Path to save processed video")
    p.add_argument("--model_path", "-m", type=str, 
                   default="saved_models/AntiSpoofing_bin_1.5_128.onnx", 
                   help="Path to ONNX model")
    p.add_argument("--threshold", "-t", type=check_zero_to_one, default=0.5, 
                   help="real face probability threshold above which the prediction is considered true")
    args = p.parse_args()
    
    face_detector = YOLOv5('saved_models/yolov5s-face.onnx')
    anti_spoof = AntiSpoof(args.model_path)

    # Create a video capture object
    if args.input:  # file
        vid_capture = cv2.VideoCapture(args.input)
    else:           # webcam
        vid_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    frame_width = int(vid_capture.get(3))
    frame_height = int(vid_capture.get(4))
    frame_size = (frame_width, frame_height)
    print('Frame size  :', frame_size)

    if not vid_capture.isOpened():
        print("Error opening a video stream")
    # Reading fps and frame rate
    else:
        fps = vid_capture.get(5)    # Get information about frame rate
        print('Frame rate  : ', fps, 'FPS')
        if fps == 0:
            fps = 15
        # frame_count = vid_capture.get(7)    # Get the number of frames
        # print('Frames count: ', frame_count) 
    
    # videowriter
    if args.output: 
        output = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'XVID'), fps, frame_size)
    print("Video is processed. Press 'Q' or 'Esc' to quit")
    
    # process frames
    rec_width = max(1, int(frame_width/240))
    txt_offset = int(frame_height/50)
    txt_width = max(1, int(frame_width/480))
    while vid_capture.isOpened():
        ret, frame = vid_capture.read()
        if ret:
            # predict score of Live face
            pred = make_prediction(frame, face_detector, anti_spoof)
            # if face is detected
            if pred is not None:
                (x1, y1, x2, y2), label, score = pred
                if label == 0:
                    if score > args.threshold:
                        res_text = "REAL      {:.2f}".format(score)
                        color = COLOR_REAL
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

                    else: 
                        res_text = "unknown"
                        color = COLOR_UNKNOWN
                else:
                    res_text = "FAKE      {:.2f}".format(score)
                    color = COLOR_FAKE
                    
                # draw bbox with label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, rec_width)
                cv2.putText(frame, res_text, (x1, y1-txt_offset), 
                            cv2.FONT_HERSHEY_COMPLEX, (x2-x1)/250, color, txt_width)
            
            if args.output:
                output.write(frame)
            
            # if video captured from webcam
            if not args.input:
                cv2.imshow('Face AntiSpoofing', frame)
                key = cv2.waitKey(20)
                if (key == ord('q')) or key == 27:
                    break
        else:
            print("Streaming is Off")
            break

    # Release the video capture and writer objects
    vid_capture.release()
    output.release()
