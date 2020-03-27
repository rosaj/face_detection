from yoloface import yoloface
from mtcnn import mtcnn
from haarcascade import haarcascade
from retina_face import retina_face
from dsfd import dsfd
from S3FD import s3fd
from centerface import centerface

from yoloface.utils import *
import cv2


def do_detect(stream_path, detector):
    wind_name = 'Face Detection using ' + detector.Name
    cv2.namedWindow(wind_name, cv2.WINDOW_NORMAL)

    cap = cv2.VideoCapture(stream_path)

    while True:

        has_frame, frame = cap.read()

        # Stop the program if reached end of video
        if not has_frame:
            print('[i] ==> Done processing!!!')
            cv2.waitKey(1000)
            break

        faces = detector.detect_faces(frame)

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

        for f in faces:
            b = f.bbox
            draw_predict(frame, f.conf, b[0], b[1], b[2], b[3])

        cv2.imshow(wind_name, frame)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            print('[i] ==> Interrupted by user!')
            break

    cap.release()
    cv2.destroyAllWindows()

    print('==> All done!')
    print('***********************************************************')


if __name__ == '__main__':
    do_detect('sut_KS_48.mp4', centerface)
