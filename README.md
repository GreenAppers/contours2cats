# contours2cats

## Datasets

Prepare cat images, e.g. from:

- https://www.kaggle.com/crawford/cat-dataset
- https://www.kaggle.com/shubhamagarwal269/cat-vs-dog

into ./data/CATS*/*.jpg

## Grep for cats with [Mask_RCNN](https://github.com/matterport/Mask_RCNN)

```
mkdir cats
mkdir mask_logs
./catgrep.py
```

## Make contour drawings with [PhotoSketch](https://github.com/mtli/PhotoSketch)

Download pretrained model from https://github.com/mtli/PhotoSketch/tree/0f28087d0b7fb5d4e4e13957ad2395e7e315042a
into ./PhotoSketch/pretrained/

```
mkdir contours
./cats2contours.sh
```

## Prepare training data

```
mkdir resized-cats
mkdir resized-contours
python pix2pix-tensorflow/tools/process.py --input_dir cats     --operation resize --output_dir resized-cats
python pix2pix-tensorflow/tools/process.py --input_dir contours --operation resize --output_dir resized-contours

mkdir contours2cats
python pix2pix-tensorflow/tools/process.py \
  --input_dir resized-contours \
  --b_dir resized-cats \
  --operation combine \
  --output_dir contours2cats
```

## Train contours2cats with [pix2pix-tensorflow](https://github.com/affinelayer/pix2pix-tensorflow)

```
mkdir contours2cats_train
python pix2pix-tensorflow/pix2pix.py \
  --mode train \
  --max_epochs 200 \
  --save_freq 2000 \
  --norm_type tflite_compatible \
  --input_dir contours2cats \
  --output_dir contours2cats_train
```

## Export contours2cats model

```
mkdir contours2cats_export
python pix2pix-tensorflow/pix2pix.py \
  --mode export \
  --export_format tflite \
  --norm_type tflite_compatible \
  --checkpoint contours2cats_train \
  --output_dir contours2cats_export
```

