import cv2
import face_recognition
import numpy as np
import torch
import os

def load_known_faces():
    known_face_encodings = []
    known_face_names = []
    folder_path = "known_faces"
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".jpg") or file_name.endswith(".png"):
            image_path = os.path.join(folder_path, file_name)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(os.path.splitext(file_name)[0])
    return known_face_encodings, known_face_names

def test_webcam():
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        torch.cuda.set_device(0)
    else:
        print("CUDA not available. Using CPU.")
    
    known_face_encodings, known_face_names = load_known_faces()
    video_capture = cv2.VideoCapture(0)
    
    # Increase the resolution
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Increase width
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Increase height
    video_capture.set(cv2.CAP_PROP_FPS, 30)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
        face_locations = face_recognition.face_locations(small_frame, model="hog")
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.75, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('Face Recognition System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_webcam()
