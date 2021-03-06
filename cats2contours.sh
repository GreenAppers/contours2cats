python ./PhotoSketch/test_pretrained.py \
    --name pretrained \
    --dataset_mode test_dir \
    --dataroot cats \
    --results_dir contours \
    --checkpoints_dir ./PhotoSketch \
    --model pix2pix \
    --which_direction AtoB \
    --norm batch \
    --input_nc 3 \
    --output_nc 1 \
    --which_model_netG resnet_9blocks \
    --no_dropout \
