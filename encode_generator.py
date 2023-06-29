import cv2
import face_recognition
import pickle
import os

# importing the images of people
folder_path = 'Images'
path_list = os.listdir(folder_path)

# An array to store the images
images = []

# this array stores the student IDs
student_id = []
for path in path_list:
    images.append(cv2.imread(os.path.join(folder_path, path)))
    student_id.append(os.path.splitext(path)[0])


# this function will encode all the images and store them inside another list known as encode list
def createEncoding(images_list):
    print("Encoding started...")

    encodings_list = []
    for image in images_list:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(image)[0]
        encodings_list.append(encode)

    return encodings_list


# passing the list which contain the images in order to record
known_encode_list = createEncoding(images)
print("Encoding complete!")

known_encode_list_with_id = [known_encode_list, student_id]

file = open("Encode_file.p", "wb")
pickle.dump(known_encode_list_with_id, file)
file.close()
print("File has been saved!")
