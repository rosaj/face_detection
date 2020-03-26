# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import cv2
import time
import numpy as np
from PIL import Image

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
    # list_bbox_tlbr = []
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= thresh:
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            score = detections[0, i, j, 0].cpu().numpy()

            det_faces.append(DetFace(float(score), (pt[0], pt[1], pt[2], pt[3])))
            # list_bbox_tlbr.append([pt[1], pt[0], pt[3], pt[2], float(score)])
            j += 1

    return det_faces


"""

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import cv2
import time
import numpy as np
from PIL import Image

from S3FD.data.config import cfg
from S3FD.s3fd_model import build_s3fd
from S3FD.utils.augmentations import to_chw_bgr

torch.set_default_tensor_type('torch.FloatTensor')


def detect_image(net, img_orig, thresh):
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
        y = net(x)
    detections = y.data

    img = img_orig.copy()
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])

    list_bbox_tlbr = []
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= thresh:
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            score = detections[0, i, j, 0].cpu().numpy()
            # left_up, right_bottom = (pt[0], pt[1]), (pt[2], pt[3])
            list_bbox_tlbr.append([pt[1], pt[0], pt[3], pt[2], float(score)])
            j += 1

    return list_bbox_tlbr


if __name__ == '__main__':

    net = build_s3fd('test', cfg.NUM_CLASSES)

    net.load_state_dict(torch.load('S3FD/weights/sfd_face.pth', map_location=torch.device('cpu')))
    net.eval()

    # vc = cv2.VideoCapture(args.input_video)

    # i = 0
    # while True:
    #     i += 1
    #     img = vc.read()[1]
    #     if img is None:
    #         break
    #     if i%2 == 0:
    #         continue
    #     show = img.copy()

    #     list_bbox_tlbr = detect_image(net, img, args.thresh)

    #     for bbox in list_bbox_tlbr:
    #         t,l,b,r,conf = bbox
    #         cv2.rectangle(show, (l,t), (r,b), (0, 0, 255), 2)
    #         conf = "{:.2f}".format(conf)
    #         point = (int(l), int(t - 5))
    #         cv2.putText(show, conf, point, cv2.FONT_HERSHEY_SIMPLEX,
    #                    0.6, (0, 255, 0), 1, lineType=cv2.LINE_AA)

    #     cv2.imshow('show', show)
    #     key = cv2.waitKey(1)
    #     if key == 27:
    #         break

    bgr_img = cv2.imread('./first.jpg', 1)
    print(bgr_img.shape)

    # detection
    for idx in range(10):
        (h, w) = bgr_img.shape[:2]

        list_bbox_tlbr = detect_image(net, bgr_img, 0.2)

    # draw rectangle bbox
    for bb in list_bbox_tlbr:
        (t, l, b, r, conf) = bb
        cv2.rectangle(bgr_img, (l, t), (r, b), (0, 255, 0), 2)
        conf = "{:.2f}".format(conf)
        point = (int(l), int(t - 5))
        cv2.putText(bgr_img, conf, point, cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 1, lineType=cv2.LINE_AA)

    cv2.namedWindow('show', 0)
    cv2.imshow('show', bgr_img)
    cv2.waitKey()
"""
