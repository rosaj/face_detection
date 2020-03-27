
# Implementation from https://github.com/hukkelas/DSFD-Pytorch-Inference
# Original code and model: https://github.com/TencentYoutuResearch/FaceDetection-DSFD


from dsfd import detect

from common.det_face import DetFace

Name = 'DSFD'


def __load_model():
    return detect.DSFDDetector()


__model = __load_model()


def detect_faces(frame, thresh=0.1):

    faces = __model.detect_face(frame, thresh)

    det_faces = [DetFace(b[4], (b[0], b[1], b[2], b[3])) for b in faces]
    return det_faces
