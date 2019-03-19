#!/usr/bin/env python

import os
import sys
import glob
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from pprint import pprint

# Root directory of the project
ROOT_DIR = os.path.abspath('./Mask_RCNN')

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, 'samples/coco/'))  # To find local version
import coco

MODEL_DIR = os.path.join(ROOT_DIR, '../mask_logs')
COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'mask_rcnn_coco.h5')

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def mask_background(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == 0, 255, image[:, :, c])
    return image


def main():
    config = InferenceConfig()
    config.display()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode='inference', model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    # First 81 COCO Class names
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']

    cat_class_id = class_names.index('cat')

    file_names = glob.glob(os.path.join(ROOT_DIR, '../data/CAT*/*.jpg'))
    total_files = len(file_names)

    index = 0
    for file_name in file_names:
        image = skimage.io.imread(file_name)

        # Run detection
        results = model.detect([image], verbose=1)
        if len(results) < 1:
            continue

        r = results[0]
        if len(r['class_ids']) < 1:
            continue

        for i in range(0, 1): #len(r['class_ids'])):
            if r['scores'][i] < 0.95:
                break
            if r['class_ids'][i] != cat_class_id:
                continue

            #print('Got cat ' + str(r['rois'][i]))
            (y1, x1, y2, x2) = r['rois'][i]
            mask_background(image, r['masks'][:, :, i], [1, 0, 0])
            cropped = image[y1:y2, x1:x2]

            index = index + 1
            out_file_name = 'cats/' + '{:08d}'.format(index) + '.png'
            skimage.io.imsave(out_file_name, cropped)
            print(str(index) + '/' + str(total_files) + ': ' + file_name + ' -> ' + out_file_name)

main()
