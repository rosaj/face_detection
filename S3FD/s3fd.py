# -*- coding:utf-8 -*-

# https://github.com/Team-Neighborhood/awesome-face-detection/tree/master/S3FD


import torch
"""
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import time
import numpy as np
from PIL import Image
"""
import cv2

from S3FD.data.config import cfg
from S3FD.s3fd_model import build_s3fd
from S3FD.utils.augmentations import to_chw_bgr

from common.det_face import DetFace


torch.set_default_tensor_type('torch.FloatTensor')

Name = 'S3FD'


def __load_model():
    net = build_s3fd('test', cfg.NUM_CLASSES)

    net.load_state_dict(torch.load('./S3FD/weights/sfd_face.pth', map_location=torch.device('cpu')))
    net.eval()
    return net


__model = __load_model()


def detect_faces(frame, thresh=0.2):
    img_orig = frame
    img = cv2.cvtColor(img_orig.copy(), cv2.COLOR_BGR2RGB)

    # height, width, _ = img.shape

    # max_im_shrink = np.sqrt(
    #     1700 * 1200 / (img.shape[0] * img.shape[1]))

    # image = cv2.resize(img, None, None, fx=max_im_shrink,
    #                    fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)

    # image = cv2.resize(img, (640, 640))
    # image = cv2.resize(image, None, fx=1 / 8, fy=1 / 8)
    # print (image.shape)

    image = img
    x = to_chw_bgr(image)
    x = x.astype('float32')
    x -= cfg.img_mean
    x = x[[2, 1, 0], :, :]

    x = torch.from_numpy(x).unsqueeze(0)

    with torch.no_grad():
        y = __model(x)
    detections = y.data

    img = img_orig.copy()
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])

    det_faces = []
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= thresh:
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            score = detections[0, i, j, 0].cpu().numpy()

            det_faces.append(DetFace(float(score), (pt[0], pt[1], pt[2], pt[3])))
            j += 1

    return det_faces

