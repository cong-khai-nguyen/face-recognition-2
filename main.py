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

def get_optimal_font_scale(text, width):
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
        new_width = textSize[0][0]
        if (new_width <= width):
            return scale/13
    return 1


faces = get_encoded_faces()
faces_encoded = list(faces.values())
known_face_names = list(faces.keys())

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frameS = cv2.resize(frame, (0,0), None, fx = 0.25, fy = 0.25)
    # frameS = cv2.cvtColor(frameS, cv2.COLOR_BGR2RGB)

    facesCurrFrame = fr.face_locations(frameS)
    encodesCurrFrame = fr.face_encodings(frameS, facesCurrFrame)

    face_names = []
    for encodeFace,faceLoc in zip(encodesCurrFrame, facesCurrFrame):
        name = "Unknown"
        matches = fr.compare_faces(faces_encoded, encodeFace)
        face_distances = fr.face_distance(faces_encoded, encodeFace)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index].upper()

        face_names.append(name)
        for (top, right, bottom, left), name in zip(facesCurrFrame, face_names):
            top, right, bottom, left = top*4, right*4, bottom*4, left*4
            # Draw a box around the face
            cv2.rectangle(frame, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)

            #Draw label
            cv2.rectangle(frame, (left-20, bottom-15), (right+20, bottom+20), (255,0,0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX

            cv2.putText(frame, name, (left-20, bottom+15), font, get_optimal_font_scale(name, right-left+40), (255,255,255), 2)


    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()