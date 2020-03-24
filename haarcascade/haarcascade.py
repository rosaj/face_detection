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

    det_faces = [DetFace(1, (x, y, x + w, y + h))  for (x, y, w, h) in faces]
    return det_faces


"""
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)


def detect_faces(stream_path, conf_threshold=0.00, nms_threshold=0.00):

    face_cascade = cv2.CascadeClassifier('./haarcascade/haarcascade_frontalface_default.xml')

    wind_name = 'Face Detection using haarcascade'
    cv2.namedWindow(wind_name, cv2.WINDOW_NORMAL)

    cap = cv2.VideoCapture(stream_path)

    while True:

        has_frame, frame = cap.read()

        # Stop the program if reached end of video
        if not has_frame:
            print('[i] ==> Done processing!!!')
            cv2.waitKey(1000)
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,   # Parameter specifying how much the image size is reduced at each image scale.
            minNeighbors=3,     # Parameter specifying how many neighbors each candidate rectangle should have to retain it
            minSize=(5, 5),     # Minimum possible object size. Objects smaller than that are ignored
            maxSize=None,       # Maximum possible object size. Objects larger than that are ignored. If `maxSize == minSize` model is evaluated on single scale.
            flags=cv2.CASCADE_SCALE_IMAGE
        )

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

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_YELLOW, 2)

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
