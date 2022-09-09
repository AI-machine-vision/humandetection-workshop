import cv2
import argparse
# import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from utils.priors import *
from model.ssd import ssd
import matplotlib.pyplot as plt
from utils.post_processing import post_process
from matplotlib.image import imread
import os
from utils.preprocessing import prepare_for_prediction
import argparse

np.random.seed(42)
COLORS = [list(np.random.random(size=3) * 256) for i in range(20)]
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'] 

def parse_opt():
    parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Keras')
    parser.add_argument('-fill', action='store_true', help='fill bbox')
    parser.add_argument('-video', default=None,
                    help='Directory of video')
    return parser.parse_args()


def draw_bboxes(bboxes, frame, labels=None, fill=False, IMAGE_SIZE=[300, 300]):
    if np.max(bboxes) < 20:
        bboxes[:, [0,2]] = bboxes[:, [0,2]]*IMAGE_SIZE[1]
        bboxes[:, [1,3]] = bboxes[:, [1,3]]*IMAGE_SIZE[0]
    obj = []
    overlay = frame.copy()
    alpha = 0.1
    for i, bbox in enumerate(bboxes):
        ind = int(labels[i] - 1)
        ind = 14 if CLASSES[int(labels[i] - 1)] == 'person' else 12
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                             (int(bbox[2]), int(bbox[3])), COLORS[ind], 2)
        if fill:
            cv2.rectangle(overlay, (int(bbox[0]), int(bbox[1])), 
                                (int(bbox[2]), int(bbox[3])), COLORS[ind], -1)
        if labels is not None:
            label = 'Human' if CLASSES[int(labels[i] - 1)] == 'person' else 'Other'
            cv2.putText(frame, label, 
                               (int(bbox[0]+0.5), int(bbox[1]-10)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                               COLORS[ind], 2)
        obj.append([label,int(bbox[0]), int(bbox[1])])
    if fill:
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return np.asarray(obj), frame


def main(opt):
    # input
    if opt.video:
        cap = cv2.VideoCapture(opt.video)
    else:
        cap = cv2.VideoCapture(0)

    # Check whether user selected camera is opened successfully.
    if not (cap.isOpened()):
        print("Could not open video device")
        cap.release()
    else:
        print("Welcome to Human detection")
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # load weight
    IMAGE_SIZE = [300, 300]
    checkpoint_filepath = './checkpoints/efficientnetb0_SSD.h5'

    iou_threshold = 0.5
    center_variance = 0.1
    size_variance = 0.2

    specs = [
                    SSDSpec(38, 8, SSDBoxSizes(30, 60), [2]),
                    SSDSpec(19, 16, SSDBoxSizes(60, 111), [2, 3]),
                    SSDSpec(10, 32, SSDBoxSizes(111, 162), [2, 3]),
                    SSDSpec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
                    SSDSpec(3, 100, SSDBoxSizes(213, 264), [2]),
                    SSDSpec(1, 300, SSDBoxSizes(264, 315), [2])
            ]

    priors = generate_ssd_priors(specs, IMAGE_SIZE[0])
    target_transform = MatchPrior(priors, center_variance, size_variance, iou_threshold)

    print("Building SSD Model with EfficientNet{0} backbone..".format("B0"))
    model = ssd("B0", pretrained=False)

    print("Loading Checkpoint..")
    if checkpoint_filepath is not None:
        print("Loading Checkpoint..", checkpoint_filepath)
        model.load_weights(checkpoint_filepath)
    else:
        print("Training from with only base model pretrained on imagenet")

    try:
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            # convert to Keras tensor
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (300,300))
            img = img/255.0
            img = np.reshape(img,(1,300,300,3))

            # Our operations on the frame come here
            pred = model.predict(img, verbose=1)
            predictions = post_process(pred, target_transform, confidence_threshold=0.4)

            pred_boxes, pred_scores, pred_labels = predictions[0]
            if pred_boxes.size > 0:
                obj, frame = draw_bboxes(pred_boxes, frame , labels=pred_labels, fill=opt.fill, IMAGE_SIZE=(height, width))
            
                cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        print('Stopped by keyboard interrupt')
        cap.release()

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
