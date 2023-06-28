import cv2



# start capturing video footage from the webcam
cap = cv2.VideoCapture(0)

# the width and height of the graphic that is being captured 
cap.set(3,1280)
cap.set(4,720)


while True:
    success, img = cap.read()
    cv2.imshow("Cam feed", img)
    cv2.waitKey(1)
