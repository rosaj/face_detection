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


"""
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)


def detect_faces(stream_path, conf_threshold=0.00, nms_threshold=0.00):
    wind_name = 'Face Detection using MTCNN'
    cv2.namedWindow(wind_name, cv2.WINDOW_NORMAL)

    cap = cv2.VideoCapture(stream_path)

    # Create face detector
    mtcnn = MTCNN(keep_all=True,  # detect multiple faces per image
                  post_process=False,
                  # Whether or not to post process images tensors before returning (False prevents normalization)
                  min_face_size=10,  # Minimum face size to search for (default: {20})
                  # thresholds=[0.3, 0.4, 0.5],  # MTCNN face detection thresholds (default: {[0.6, 0.7, 0.7]})
                  # factor=0.4,  # Factor used to create a scaling pyramid of face sizes. (default: {0.709})
                  device='cpu')

    while True:

        has_frame, frame = cap.read()

        c_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        c_frame = Image.fromarray(c_frame)

        # Detect face
        faces, probs = mtcnn.detect(c_frame)
        faces = [] if faces is None else faces

        # Stop the program if reached end of video
        if not has_frame:
            print('[i] ==> Done processing!!!')
            cv2.waitKey(1000)
            break

        print('[i] ==> # detected faces: {}'.format(len(faces)))
        print('#' * 60)

        # initialize the set of information we'll displaying on the frame
        info = [
            ('number of faces detected', '{}'.format(len(faces)))
        ]

        for (i, (txt, val)) in enumerate(info):
            text = '{}: {}'.format(txt, val)
            cv2.putText(frame, text, (10, (i * 20) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)

        for b in faces:
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), COLOR_YELLOW, 2)

        cv2.imshow(wind_name, frame)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            print('[i] ==> Interrupted by user!')
            break

    cap.release()
    cv2.destroyAllWindows()

    print('==> All done!')
    print('***********************************************************')
"""
