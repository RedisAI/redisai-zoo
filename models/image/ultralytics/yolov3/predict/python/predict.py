# This script runs YOLO v3 inside RedisAI.
# It can be ran at any location, it doesn't depend on code
# in the ultralytics/yolov3 repository.
# To this end, some of the utility functions (mainly related to
# non-maximum suppression) have been included here.
# Note that they could also be ported to RedisAI SCRIPT quite
# easily, so the client doesn't need to perform Numpy operations.

import argparse
import numpy as np
import cv2
import os
import sys
import redisai


DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 6379


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_LINEAR)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image

    return canvas


def xywh2xyxy(x):
    # Transform box coordinates from [x, y, w, h] to [x1, y1, x2, y2] (where xy1=top-left, xy2=bottom-right)
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0] = np.clip(boxes[:, 0], 0, img_shape[1])  # x1
    boxes[:, 1] = np.clip(boxes[:, 1], 0, img_shape[0])  # y1
    boxes[:, 2] = np.clip(boxes[:, 2], 0, img_shape[1])  # x2
    boxes[:, 3] = np.clip(boxes[:, 3], 0, img_shape[0])  # y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.transpose())
    area2 = box_area(box2.transpose())

    inter = (np.minimum(box1[:, None, 2:], box2[:, 2:]) - np.maximum(box1[:, None, :2], box2[:, :2])).clip(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, classes=None):
    """
    Performs  Non-Maximum Suppression on inference results
    Returns detections with shape:
        nx6 (x1, y1, x2, y2, conf, cls)
    """

    # Box constraints
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height

    nc = prediction.shape[1] - 5  # number of classes
    output = [None] * len(prediction)

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply conf constraint
        x = x[x[:, 4] > conf_thres]

        # Apply width-height constraint
        x = x[((x[:, 2:4] > min_wh) & (x[:, 2:4] < max_wh)).all(1)]

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[..., 5:] *= x[..., 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        conf = x[:, 5:].max(axis=1)
        j = x[:, 5:].argmax(axis=1)
        x = np.concatenate((box, conf[:, None], j.astype(np.float32)[:, None]), axis=1)

        # Filter by class
        if classes:
            x = x[(j.reshape(-1, 1) == np.array(classes)).any(1)]

        # Apply finite constraint
        if not np.isfinite(x).all():
            x = x[np.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Batched NMS
        c = x[:, 5]  # classes
        boxes, scores = x[:, :4].copy() + c.reshape(-1, 1) * max_wh, x[:, 4]  # boxes (offset by class), scores
        iou = np.triu(box_iou(boxes, boxes), k=1)  # upper triangular iou matrix
        i = iou.max(0, keepdims=True)[0] < iou_thres

        output[xi] = x[i]

    return output


def predict(r, key, img, orig_shape, names, conf_thresh, iou_thresh):
    # Put channels first
    img = np.transpose(img, (2, 0, 1))

    # Add batch dimension of 1, convert to float32 and normalize
    img = img[None, :].astype(np.float32) / 255.0

    dag = r.dag()
    dag.tensorset('in', img)
    dag.modelrun(key, ['in'], ['out'])
    dag.tensorget('out')
    pred = dag.run()[-1]

    pred = non_max_suppression(pred, conf_thresh, iou_thresh, classes=None)

    out = []
    for i, det in enumerate(pred):  # detections per image
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orig_shape).round()

            # Print results
            for *xyxy, conf, cls in det:
                out.append((names[int(cls)], conf, xyxy))

    return out


def predict_from_files(filenames, key,
                       img_size=512, names_file='data/coco.names',
                       conf_thresh=0.3, iou_thresh=0.6,
                       host=DEFAULT_HOST,
                       port=DEFAULT_PORT):

    names = load_classes(names_file)

    r = redisai.Client(host=host, port=port)

    outputs = []

    for filename in filenames:
        img = cv2.imread(filename)
        img_lb = letterbox_image(img, (img_size, img_size))

        out = predict(r, key, img_lb,
                      orig_shape=img.shape[:2],
                      names=names,
                      conf_thresh=conf_thresh,
                      iou_thresh=iou_thresh)

        outputs.append(out)

    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filenames', type=str, default='', help='input files')
    parser.add_argument('--key', type=str, default='yolov3', help='model key')
    parser.add_argument('--names', type=str, default='coco.names', help='file containing Coco label names')
    parser.add_argument('--prefix', type=str, default='', help='key prefix')
    parser.add_argument('--host', type=str, default='120.0.0.1', help='Redis host')
    parser.add_argument('--port', type=int, default=6379, help='Redis port')
    opt = parser.parse_args()

    if not opt.filenames:
        print("No filenames specified")
        sys.exit(0)

    filenames = opt.filenames.split(',')

    ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
    names = os.path.join(ROOT_DIR, '..', '..', opt.names)

    out = predict_from_files(filenames, opt.key, names_file=names)

    print(out)
