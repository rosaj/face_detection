import insightface
import cv2
import numpy as np

from common.det_face import DetFace

# https://github.com/deepinsight/insightface/tree/master/RetinaFace
# RetinaFace and ArcFace

Name = 'RetinaFace'
Recognition = False  # Whether or not to do one-shot recognition

# Minimal threshold similarity. If no faces with higher similarity, a new face is added to recognized faces list
FACE_CONF_THRESHOLD = 0.2


def __load_model():
    fa = insightface.app.FaceAnalysis()
    fa.prepare(-1)
    return fa


__model = __load_model()


def compute_sim(face1, face2):
    """
    Cosing similarity between two faces
    :param face1: First face
    :param face2: Second face
    :return: similarity as float
    """
    return np.dot(face1.embedding, face2.embedding) / (face1.embedding_norm * face2.embedding_norm)


__recognized_faces = []


def find_face(face):
    """
    Finds most similar face in the already recognized list of faces
    :param face: Face which needs to be recognized
    :return: index and highest similarity of recognized face. Returns None and 0 if no faces are registered
    """
    if len(__recognized_faces) == 0:
        return None, 0

    recogs = np.array([compute_sim(face, f2) for f2 in __recognized_faces])
    # Get the index with max similarity
    ind = np.argmax(recogs)
    # print(max(dets))
    return ind, recogs[ind]


def detect_faces(frame, thresh=0.1):
    faces = __model.get(frame,
                        det_thresh=thresh,
                        det_scale=2.0
                        )

    det_faces = []

    for face in faces:
        df = DetFace(face.det_score, face.bbox)
        det_faces.append(df)

        if Recognition:
            ind, conf = find_face(face)

            if conf < FACE_CONF_THRESHOLD:
                ind = len(__recognized_faces)
                __recognized_faces.append(face)
                conf = 1

            df.name = 'P{} ({:.2f})'.format(ind, conf)

    return det_faces
