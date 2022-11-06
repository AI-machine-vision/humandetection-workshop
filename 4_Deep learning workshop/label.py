import tensorflow as tf
from utils.priors import *
from model.ssd import ssd
import matplotlib.pyplot as plt
from utils.post_processing import post_process
from matplotlib.image import imread
import os
from utils.preprocessing import prepare_for_prediction
import argparse
import time

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
parser.add_argument('--checkpoint', default='./checkpoints/efficientnetb0_SSD.h5',
                    help='Directory of checkpoint models (./checkpoints/efficientnetb0_SSD.h5)')
args = parser.parse_args()


np.random.seed(42)
COLORS = [list(np.random.random(size=3)) for i in range(20)]
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'] 



def draw_bboxes(bboxes, ax, labels=None, IMAGE_SIZE=[300, 300]):
    # image = (im - np.min(im))/np.ptp(im)
    # print(image.shape)
    if np.max(bboxes) < 10:
        bboxes[:, [0,2]] = bboxes[:, [0,2]]*IMAGE_SIZE[1]
        bboxes[:, [1,3]] = bboxes[:, [1,3]]*IMAGE_SIZE[0]
    for i, bbox in enumerate(bboxes):
        ind = int(labels[i] - 1)
        ind = 1 if CLASSES[int(labels[i] - 1)] == 'person' else 0
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor=COLORS[ind], facecolor='none')
        # ax.add_patch(rect)
        ax.add_artist(rect)
        # print(int(bbox[-1]))
        if labels is not None:
            label = 'Human' if CLASSES[int(labels[i] - 1)] == 'person' else 'Other'
            ax.text(bbox[0]+0.5,bbox[1]+0.5, label,  fontsize=20,
                horizontalalignment='left', verticalalignment='top', bbox=dict(facecolor=COLORS[ind], alpha=0.4))

def write_labels(bboxes, labels=None, IMAGE_SIZE=[300, 300], name=''):
    if np.max(bboxes) < 10:
        bboxes[:, [0,2]] = bboxes[:, [0,2]]*IMAGE_SIZE[1]
        bboxes[:, [1,3]] = bboxes[:, [1,3]]*IMAGE_SIZE[0]
    with open(name, 'a') as txt:
        for i, bbox in enumerate(bboxes):
            # bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1]
            if labels is not None:
                if CLASSES[int(labels[i] - 1)] == 'person':
                    txt.write('{} {} {} {} {}'.format(int(labels[i] - 1), 
                                                    bbox[0], bbox[1], bbox[2], bbox[3]))



if __name__ == '__main__':
    IMAGE_SIZE = [300, 300]
    BATCH_SIZE = 16
    MODEL_NAME = "B0"
    checkpoint_filepath = args.checkpoint
    
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

    print("Building SSD Model with EfficientNet{0} backbone..".format(MODEL_NAME))
    model = ssd(MODEL_NAME, pretrained=False)

    print("Loading Checkpoint..")
    if checkpoint_filepath is not None:
        print("Loading Checkpoint..", checkpoint_filepath)
        model.load_weights(checkpoint_filepath)
    else:
        print("Training from with only base model pretrained on imagenet")

    for dir_name in os.listdir('../raw data/'):
        if dir_name in ['01_champ', '02_art', '03_ter', '04_tar', '05_plug', '06_chain']:
            continue
        print(dir_name)
        INPUT_DIR = '../raw data/{}/images'.format(dir_name)
        OUTPUT_DIR = '../raw data/{}/images2label'.format(dir_name)
        OUTPUT_DIR_LABEL = '../raw data/{}/labels'.format(dir_name)
        if not os.path.exists(OUTPUT_DIR):
            os.mkdir(OUTPUT_DIR)
        if not os.path.exists(OUTPUT_DIR_LABEL):
            os.mkdir(OUTPUT_DIR_LABEL)

        dataset = tf.data.Dataset.list_files(INPUT_DIR + '/*', shuffle=False)
        filenames = list(dataset.as_numpy_iterator())
        dataset = dataset.map(prepare_for_prediction)
        dataset = dataset.batch(BATCH_SIZE)

        pred = model.predict(dataset, verbose=1)
        predictions = post_process(pred, target_transform, confidence_threshold=0.4)

        # dataset = dataset.unbatch()
        print("Prediction Complete")
        for i, path in enumerate(filenames):
            path_string = path.decode("utf-8")
            im = imread(path_string)
            # filename = path_string.split('/')[-1]
            filename = path_string.split('\\')[-1]
            fig, ax = plt.subplots(1, figsize=(15, 15))
            ax.imshow(im)
            pred_boxes, pred_scores, pred_labels = predictions[i]
            if pred_boxes.size > 0:
                draw_bboxes(pred_boxes, ax , labels=pred_labels, IMAGE_SIZE=im.shape[:2])
                write_labels(pred_boxes, labels=pred_labels, IMAGE_SIZE=im.shape[:2],
                            name = os.path.join(OUTPUT_DIR_LABEL, filename.replace('jpg', 'txt')))
            plt.axis('off')
            plt.savefig(os.path.join(OUTPUT_DIR, filename.replace('jpg', 'png')), bbox_inches='tight', pad_inches=0)
            
        print("Output is saved in", OUTPUT_DIR)
        print('Sleep 1 min')
        time.sleep(60)