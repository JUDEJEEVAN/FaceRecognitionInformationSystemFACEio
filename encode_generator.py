import cv2
import face_recognition
import pickle
import os

# importing the images of people
folder_path = 'Images'
path_list = os.listdir(folder_path)
images = []
student_id = []
for path in path_list:
    images.append(cv2.imread(os.path.join(folder_path, path)))
    student_id.append(os.path.splitext(path)[0])


def createEncoding(images_list):
    print("Encoding started...")

    encodings_list = []
    for image in images_list:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(image)[0]
        encodings_list.append(encode)

    return encodings_list


known_encode_list = createEncoding(images)
print("Encoding complete!")

