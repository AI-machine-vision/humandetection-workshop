import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from utils.priors import *
from model.ssd import ssd
from model.loss import multibox_loss
import os
from utils.preprocessing import prepare_dataset
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint
import argparse


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
# Train params
parser.add_argument('-num_epochs', default=1, type=int,
                    help='the number epochs')
args = parser.parse_args()


if __name__ == '__main__':
    DATASET_DIR = './dataset'
    IMAGE_SIZE = [300, 300]
    BATCH_SIZE = 16
    MODEL_NAME = "B0"
    EPOCHS = args.num_epochs

    checkpoint_filepath = './checkpoints/efficientnetb0_SSD.h5'
    base_lr = 1e-3 if checkpoint_filepath is None else 1e-5

    print("Loading Data..")
    train_data = tfds.load("voc", data_dir=DATASET_DIR, split='train')

    number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
    print("Number of Training Files:", number_train)


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

    # instantiate the datasets
    training_dataset = prepare_dataset(train_data, IMAGE_SIZE, BATCH_SIZE, target_transform, train=True)

    print("Building SSD Model with EfficientNet{0} backbone..".format(MODEL_NAME))
    model = ssd(MODEL_NAME)
    steps_per_epoch = number_train // BATCH_SIZE
    print("Number of Train Batches:", steps_per_epoch)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr),
        loss = multibox_loss
    )

    if checkpoint_filepath is not None:
        print("Continuing Training from", checkpoint_filepath)
        model.load_weights(checkpoint_filepath)
    else:
        print("Training from with only base model pretrained on imagenet")

    history = model.fit(training_dataset, 
                        steps_per_epoch=steps_per_epoch, 
                        epochs=EPOCHS) 