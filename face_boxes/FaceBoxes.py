

# https://paperswithcode.com/paper/faceboxes-a-cpu-real-time-face-detector-with
# https://github.com/sfzhang15/FaceBoxes
# https://github.com/zisianw/FaceBoxes.PyTorch


import torch
import torch.backends.cudnn as cudnn
import numpy as np

from face_boxes.data import cfg
from face_boxes.layers.functions.prior_box import PriorBox
from face_boxes.utils.nms_wrapper import nms
from face_boxes.models.faceboxes import FaceBoxes
from face_boxes.utils.box_utils import decode

import cv2
from common.det_face import DetFace

Name = 'FaceBoxes'


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    # print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def __load_model():
    torch.set_grad_enabled(False)
    # net and model
    net = FaceBoxes(phase='test', size=None, num_classes=2)  # initialize detector
    net = load_model(net, './face_boxes/weights/FaceBoxesProd.pth', True)
    net.eval()
    # print('Finished loading model!')
    # print(net)
    cudnn.benchmark = True
    device = torch.device("cpu")
    net = net.to(device)

    return net, device


__model, __device = __load_model()


def detect_faces(frame, thresh=0.05, nms_thresh=0.3):
    # resize = 3
    resize = 1

    # img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = np.float32(frame)
    if resize != 1:
        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(__device)
    scale = scale.to(__device)

    loc, conf = __model(img)  # forward pass

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(__device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

    # ignore low scores
    inds = np.where(scores > thresh)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    top_k = 5000
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)

    keep = nms(dets, nms_thresh, force_cpu=True)
    dets = dets[keep, :]

    # keep top-K faster NMS
    keep_top_k = 750
    dets = dets[:keep_top_k, :]

    det_faces = [DetFace(b[4], (b[0], b[1], b[2], b[3])) for b in dets]

    return det_faces
