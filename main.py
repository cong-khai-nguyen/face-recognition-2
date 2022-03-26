import cv2
import numpy as np
import os
import face_recognition as fr

# function to encodes all the faces in the faces folder
# return a dict of (name, image encoded) -> Ex: bill gate : [......]
def get_encoded_faces():
    encoded = {}

    for dirpath, dnames, file_names in os.walk("./media/faces"):

        for f in file_names:
            if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg"):
                face = fr.load_image_file("media/faces/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded