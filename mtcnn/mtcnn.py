from facenet_pytorch import MTCNN
import cv2
from PIL import Image
from common.det_face import DetFace

Name = 'MTCNN'


def __load_model():
    # Create face detector
    mtcnn = MTCNN(keep_all=True,  # detect multiple faces per image
                  post_process=False,
                  # Whether or not to post process images tensors before returning (False prevents normalization)
                  min_face_size=10,  # Minimum face size to search for (default: {20})
                  # thresholds=[0.3, 0.4, 0.5],  # MTCNN face detection thresholds (default: {[0.6, 0.7, 0.7]})
                  # factor=0.4,  # Factor used to create a scaling pyramid of face sizes. (default: {0.709})
                  device='cpu')

    return mtcnn


__model = __load_model()


def detect_faces(frame):
    c_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    c_frame = Image.fromarray(c_frame)

    # Detect face
    faces, probs = __model.detect(c_frame)
    faces = [] if faces is None else faces

    det_faces = [DetFace(conf, bbox) for conf, bbox in zip(probs, faces)]
    return det_faces
