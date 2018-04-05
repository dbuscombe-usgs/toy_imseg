IMAGE_SIZE=96

python2 retrain.py --image_dir train/tile_$IMAGE_SIZE \
    --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/classification/1 \
    --how_many_training_steps 1000 \
    --output_labels labels.txt --output_graph mobilenetv2_"{$IMAGE_SIZE}"_graph.pb

rm -rf /tmp/bottleneck
rm -rf /tmp/checkpoint
