#! /usr/bin/env python3
import cv2
import numpy as np
import time
from src import masknet
import darknet
from scipy.misc import imresize
import os
import tensorflow as tf
import keras.backend as K


def convertBack(x, y, w, h):
    """
      bbox coordinate transform
    """
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img, width, height):

    rois = []
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x*(1920/width)), float(y*(1080/height)), float(w*(1920/width)), float(h*(1080/height)))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
        rois.append([np.float32(ymin / 1080), np.float32(xmin / 1920), np.float32(ymax / 1080), np.float32(xmax / 1920)])

    num_true_rois = len(rois)

    for _ in range(masknet.my_num_rois - len(rois)):
        rois.append([np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(0.0)])
    return img, np.array(rois), num_true_rois


def YOLO():

    configPath = "./yolov3_best.cfg"
    weightPath = "./yolov3_best_final.weights"
    metaPath = "./cfg/maskYolo.data"
    netMain = darknet.load_net_custom(configPath.encode(
        "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1

    metaMain = darknet.load_meta(metaPath.encode("ascii"))
    try:
        with open(metaPath) as metaFH:
            metaContents = metaFH.read()
            import re
            match = re.search("names *= *(.*)$", metaContents,
                              re.IGNORECASE | re.MULTILINE)
            if match:
                result = match.group(1)
            else:
                result = None
            try:
                if os.path.exists(result):
                    with open(result) as namesFH:
                        namesList = namesFH.read().strip().split("\n")
                        altNames = [x.strip() for x in namesList]
            except TypeError:
                pass
    except Exception:
        pass
	
    model = masknet.create_model()
    model.summary()
    model.load_weights("yolov3mask_28.hdf5")
    cap = cv2.VideoCapture("test_video/4.mov")
    print("Starting the YOLO")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain), 3)
    n = 0
    while True:
        ret, frame_read = cap.read()
        n = n + 1
        if frame_read is None:
            print('\nEnd of Video')
            break
        prev_time = time.time()
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.5)
        tensor_11 = darknet.get_tensor(netMain, 11)
        tensor_36 = darknet.get_tensor(netMain, 36)
        tensor_61 = darknet.get_tensor(netMain, 61)
        tensor_74 = darknet.get_tensor(netMain, 74)
        tensor_11 = np.ctypeslib.as_array(tensor_11, [1, 128, 80, 80]).transpose(0, 2, 3, 1)  # B,H,W,C
        tensor_36 = np.ctypeslib.as_array(tensor_36, [1, 256, 40, 40]).transpose(0, 2, 3, 1)
        tensor_61 = np.ctypeslib.as_array(tensor_61, [1, 512, 20, 20]).transpose(0, 2, 3, 1)
        tensor_74 = np.ctypeslib.as_array(tensor_74, [1, 1024, 10, 10]).transpose(0, 2, 3, 1)
        for im, single_out in zip([frame_rgb], detections):
            image, rois, true_rois = cvDrawBoxes(detections, im, darknet.network_width(netMain), darknet.network_height(netMain))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masknet_inp = rois[np.newaxis, ...]
            masks = model.predict([tensor_11, tensor_36, tensor_61, tensor_74, masknet_inp])
            masks = masks[0, :true_rois, :, :, 0]

            for i in range(true_rois):
                roi = rois[i]
                mask = masks[i]

                y1, x1, y2, x2 = roi
                if y1 > 1:
                    y1 = 1
                if y1 < 0:
                    y1 = 0
                if y2 > 1:
                    y2 = 1
                if y2 < 0:
                    y2 = 0
                if x1 > 1:
                    x1 = 1
                if x1 < 0:
                    x1 = 0
                if x2 > 1:
                    x2 = 1
                if x2 < 0:
                    x2 = 0

                left = int(x1 * 1920)
                top = int(y1 * 1080)
                right = int(x2 * 1920)
                bot = int(y2 * 1080)

                mask = imresize(mask, (bot - top, right - left), interp='bilinear').astype(np.float32) / 255.0
                mask2 = np.where(mask >= 0.5, 1, 0).astype(np.uint8)

                if (i % 3) == 0:
                    mask3 = cv2.merge((mask2 * 0, mask2 * 0, mask2 * 255))
                elif (i % 3) == 1:
                    mask3 = cv2.merge((mask2 * 0, mask2 * 255, mask2 * 0))
                else:
                    mask3 = cv2.merge((mask2 * 255, mask2 * 0, mask2 * 0))
                try:
                    image[top:bot, left:right] = cv2.addWeighted(image[top:bot, left:right], 1.0, mask3, 0.8, 0)
                except:
                    print(x1, x2, y1, y2)

            print("FPS:", round(1/(time.time()-prev_time), 2))
            cv2.namedWindow('Demo', 0)
            cv2.resizeWindow('Demo', (1536, 864))
            cv2.imshow('Demo', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    YOLO()
