import pickle
import numpy as np
import cv2
import face_recognition

# start capturing video footage from the webcam
cap = cv2.VideoCapture(0)

# the width and height of the graphic that is being captured 
cap.set(3, 640)
cap.set(4, 480)

print("Loading encoded file...")
# load the encoding file
file = open('Encode_file.p', "rb")

# to load all the encoded list into known_encode_list_with_id
known_encode_list_with_id = pickle.load(file)
file.close()
print("File loaded successfully")
# to separate the student_id and the known_encode_list from the known_encode_list_with_id
known_encode_list, student_id = known_encode_list_with_id

while True:
    # this will start reading the webcam and write it to the variable img , success
    success, img = cap.read()

    img_scaled = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    img_scaled = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2RGB)

    # this will find the face in current frame and encode
    face_in_current_frame = face_recognition.face_locations(img_scaled)
    encode_current_frame = face_recognition.face_encodings(img_scaled, face_in_current_frame)

    for face_encode, face_location in zip(encode_current_frame, face_in_current_frame):
        matches = face_recognition.compare_faces(known_encode_list, face_encode)
        face_distance = face_recognition.face_distance(known_encode_list, face_encode)
        print("Matches : ", matches)
        print("Distance : ", face_distance)

        match_index = np.argmin(face_distance)
        print("Found a match : ", student_id[match_index])

    cv2.imshow("Cam feed", img)
    cv2.waitKey(1)
