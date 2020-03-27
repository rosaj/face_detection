
from centerface.centerface_model import CenterFace

from common.det_face import DetFace

Name = 'CenterFace'


def __load_model():
    return CenterFace()


__model = __load_model()


def detect_faces(frame, thresh=0.2):
    h, w = frame.shape[:2]

    faces, _ = __model(frame, h, w, threshold=thresh)

    det_faces = [DetFace(b[4], (b[0], b[1], b[2], b[3])) for b in faces]
    return det_faces

