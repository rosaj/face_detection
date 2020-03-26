import insightface
import cv2
from common.det_face import DetFace

# https://github.com/deepinsight/insightface/tree/master/RetinaFace
# RetinaFace and ArcFace

Name = 'RetinaFace'


def __load_model():
    fa = insightface.app.FaceAnalysis()
    fa.prepare(-1)
    return fa


__model = __load_model()


def detect_faces(frame):

    faces = __model.get(frame,
                        det_thresh=0.1,
                        det_scale=2.0
                        )

    det_faces = [DetFace(face.det_score, face.bbox) for face in faces]

    return det_faces
