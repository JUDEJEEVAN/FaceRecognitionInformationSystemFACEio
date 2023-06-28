import cv2
import face_recognition

# start capturing video footage from the webcam
cap = cv2.VideoCapture(0)

# the width and height of the graphic that is being captured 
cap.set(3,1280)
cap.set(4,720)


while True:
    # this will start reading the webcam and write it to the variable img , success
    success, img = cap.read()
    cv2.imshow("Cam feed", img)
    cv2.waitKey(1)



