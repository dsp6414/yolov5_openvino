import os
import sys
import time
from argparse import ArgumentParser

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from utils.general import scale_coords, non_max_suppression

# class names
class_names = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

class_names = ["person"]
num_classes = len(class_names)

anchors = [
    [10, 13, 16, 30, 33, 23],
    [30, 61, 62, 45, 59, 119],
    [116, 90, 156, 198, 373, 326],
]  # 5s


def detect(outputs, anchors, input_size, num_classes, openvino=True):
    """
    outputs: from an onnx model or an openvino model
    """

    if openvino:
        outputs = [value for _, value in outputs.items()]

    boxs = []
    a = torch.tensor(anchors).float().view(3, -1, 2)
    anchor_grid = a.clone().view(3, 1, -1, 1, 1, 2)

    if len(outputs) == 4:
        outputs = [outputs[1], outputs[2], outputs[3]]

    for index, out in enumerate(outputs):
        out = torch.from_numpy(out)
        batch_size = 1  # using batch_size = out.shape[1] is wrong
        feature_w = out.shape[2]
        feature_h = out.shape[3]

        # Feature map corresponds to the original image zoom factor
        stride_w = int(input_size / feature_w)
        stride_h = int(input_size / feature_h)

        grid_x, grid_y = np.meshgrid(np.arange(feature_w), np.arange(feature_h))

        # cx, cy, w, h
        pred_boxes = torch.FloatTensor(out[..., :4].shape)
        pred_boxes[..., 0] = (
            torch.sigmoid(out[..., 0]) * 2.0 - 0.5 + grid_x
        ) * stride_w  # cx
        pred_boxes[..., 1] = (
            torch.sigmoid(out[..., 1]) * 2.0 - 0.5 + grid_y
        ) * stride_h  # cy
        pred_boxes[..., 2:4] = (torch.sigmoid(out[..., 2:4]) * 2) ** 2 * anchor_grid[
            index
        ]  # wh

        # pred_boxes[..., 2] = (torch.sigmoid(out[..., 2]) * 2) ** 2 * anchor_grid[index, ..., 0]  # w
        # pred_boxes[..., 3] = (torch.sigmoid(out[..., 3]) * 2) ** 2 * anchor_grid[index, ..., 1]  # h

        conf = torch.sigmoid(out[..., 4])
        pred_cls = torch.sigmoid(out[..., 5:])

        output = torch.cat(
            (
                pred_boxes.view(batch_size, -1, 4),
                conf.view(batch_size, -1, 1),
                pred_cls.view(batch_size, -1, num_classes),
            ),
            -1,
        )
        boxs.append(output)

    outputx = torch.cat(boxs, 1)

    return outputx


def resize(image, size, interpolation=cv2.INTER_CUBIC):
    shape = image.shape[0:2][::-1]
    iw, ih = shape
    w, h = size
    scale = min(w / iw, h / ih)
    ratio = scale, scale
    nw = int(round(iw * scale))
    nh = int(round(ih * scale))
    new_unpad = nw, nh

    if shape[0:2] != new_unpad:
        image = cv2.resize(image, (nw, nh), interpolation=interpolation)

    new_image = np.full((size[1], size[0], 3), 128)
    dx = (w - nw) // 2
    dy = (h - nh) // 2
    new_image[dy : dy + nh, dx : dx + nw, :] = image

    return new_image, ratio, (dx, dy)


def preprocess_cv_img(image_src, input_height, input_width, nchw_shape=True):
    in_image_src, _, _ = resize(image_src, (input_width, input_height))
    in_image_src = in_image_src[..., ::-1]
    if nchw_shape:
        in_image_src = in_image_src.transpose(
            (2, 0, 1)
        )  # Change data layout from HWC to CHW
    # in_image_src = np.ascontiguousarray(in_image_src)
    in_image_src = np.expand_dims(in_image_src, axis=0)

    img_in = in_image_src / 255.0
    return img_in


def preprocess_pil_img(image_src, input_size):
    resized, _, _ = letterbox_image(image_src, (input_size, input_size))

    img_in = np.transpose(resized, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
    img_in /= 255.0
    img_in = np.expand_dims(img_in, axis=0)

    return img_in


def letterbox_image(pil_img, size):
    iw, ih = pil_img.size
    w, h = size
    scale = min(w / iw, h / ih)
    ratio = scale, scale
    nw = int(iw * scale)
    nh = int(ih * scale)
    dw = (w - nw) // 2
    dh = (h - nh) // 2

    pil_img = pil_img.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new("RGB", size, (128, 128, 128))
    new_image.paste(pil_img, (dw, dh))
    return new_image, ratio, (dw, dh)


def display(
    detections=None,
    image_path=None,
    input_size=640,
    line_thickness=None,
    text_bg_alpha=0.0,
):
    labels = detections[..., -1]
    boxs = detections[..., :4]
    confs = detections[..., 4]

    image_src = Image.open(image_path)
    w, h = image_src.size
    image_src = np.array(image_src)

    boxs[:, :] = scale_coords((input_size, input_size), boxs[:, :], (h, w)).round()

    tl = line_thickness or round(0.002 * (w + h) / 2) + 1
    for i, box in enumerate(boxs):
        x1, y1, x2, y2 = box
        np.random.seed(int(labels[i].numpy()) + 2020)
        color = [np.random.randint(0, 255), 0, np.random.randint(0, 255)]
        cv2.rectangle(
            image_src,
            (x1, y1),
            (x2, y2),
            color,
            thickness=max(int((w + h) / 600), 1),
            lineType=cv2.LINE_AA,
        )
        label = "%s %.2f" % (class_names[int(labels[i].numpy())], confs[i])
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=1)[0]
        c2 = x1 + t_size[0] + 3, y1 - t_size[1] - 5
        if text_bg_alpha == 0.0:
            cv2.rectangle(image_src, (x1 - 1, y1), c2, color, cv2.FILLED, cv2.LINE_AA)
        else:
            alphaReserve = text_bg_alpha
            BChannel, GChannel, RChannel = color
            xMin, yMin = int(x1 - 1), int(y1 - t_size[1] - 3)
            xMax, yMax = int(x1 + t_size[0]), int(y1)
            image_src[yMin:yMax, xMin:xMax, 0] = image_src[
                yMin:yMax, xMin:xMax, 0
            ] * alphaReserve + BChannel * (1 - alphaReserve)
            image_src[yMin:yMax, xMin:xMax, 1] = image_src[
                yMin:yMax, xMin:xMax, 1
            ] * alphaReserve + GChannel * (1 - alphaReserve)
            image_src[yMin:yMax, xMin:xMax, 2] = image_src[
                yMin:yMax, xMin:xMax, 2
            ] * alphaReserve + RChannel * (1 - alphaReserve)
        cv2.putText(
            image_src,
            label,
            (x1 + 3, y1 - 4),
            0,
            tl / 3,
            [255, 255, 255],
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        print(box.numpy(), confs[i].numpy(), class_names[int(labels[i].numpy())])

    plt.imshow(image_src)
    plt.show()