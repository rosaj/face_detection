# *******************************************************************
#
# Author : Thanh Nguyen, 2018
# Email  : sthanhng@gmail.com
# Github : https://github.com/sthanhng
#
# BAP, AI Team
# Face detection using the YOLOv3 algorithm (https://github.com/sthanhng/yoloface)
#
#
# The main code of the Face detection using the YOLOv3 algorithm
# It is a YOLOv3 model trained on http://shuoyang1213.me/WIDERFACE/index.html dataset
#
# *******************************************************************


from .utils import *

Name = 'YOLOFace'


def __load_model():
    # Give the configuration and weight files for the model and load the network using them.
    net = cv2.dnn.readNetFromDarknet('./yoloface/cfg/yolov3-face.cfg',
                                     './yoloface/model-weights/yolov3-wider_16000.weights')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


__model = __load_model()


def detect_faces(frame, conf_threshold=0.00, nms_threshold=0.00):
    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (1920, 1920),
                                 [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    __model.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = __model.forward(get_outputs_names(__model))

    # Remove the bounding boxes with low confidence
    faces = post_process(frame, outs, conf_threshold, nms_threshold)

    return faces


"""


def detect_faces(stream_path, conf_threshold=0.00, nms_threshold=0.00):
    net = __load_model()

    wind_name = 'Face Detection using YOLOv3'
    cv2.namedWindow(wind_name, cv2.WINDOW_NORMAL)

    cap = cv2.VideoCapture(stream_path)

    while True:

        has_frame, frame = cap.read()

        # Stop the program if reached end of video
        if not has_frame:
            print('[i] ==> Done processing!!!')
            cv2.waitKey(1000)
            break

        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (1920, 1920),
                                     [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(get_outputs_names(net))

        # Remove the bounding boxes with low confidence
        faces = post_process(frame, outs, conf_threshold, nms_threshold)
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
