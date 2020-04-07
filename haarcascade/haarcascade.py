import cv2
from common.det_face import DetFace

Name = 'haarcascade'


def __load_model():
    face_cascade = cv2.CascadeClassifier('./haarcascade/haarcascade_frontalface_default.xml')
    return face_cascade


__model = __load_model()


def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = __model.detectMultiScale(
        gray,
        scaleFactor=1.05,  # Parameter specifying how much the image size is reduced at each image scale.
        minNeighbors=3,  # Parameter specifying how many neighbors each candidate rectangle should have to retain it
        minSize=(5, 5),  # Minimum possible object size. Objects smaller than that are ignored
        maxSize=None,
        # Maximum possible object size. Objects larger than that are ignored. If `maxSize == minSize` model is evaluated on single scale.
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    det_faces = [DetFace(1, (x, y, x + w, y + h)) for (x, y, w, h) in faces]
    return det_faces
