
declare -a arr=("96" "128" "160" "192" "224")

## now loop through the above array
for IMAGE_SIZE in "${arr[@]}"
do
   echo "Tile size: $IMAGE_SIZE"
   echo "Creating training tiles ... "
   python2 retile.py -i train -t $IMAGE_SIZE -a .9

   echo "Creating testing tiles ..."
   python2 retile.py -i test -t $IMAGE_SIZE -a .85

   cls

   echo "Retraining DCNN ..."
   python2 retrain.py --image_dir train/tile_$IMAGE_SIZE \
       --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v2_100_$IMAGE_SIZE/classification/1 \
       --how_many_training_steps 1000 \
       --output_labels labels.txt --output_graph mobilenetv2_"${IMAGE_SIZE}".pb

   rm -rf /tmp/bottleneck
   rm -rf /tmp/checkpoint

   cls
   echo "Retraining DCNN complete."

   rm -rf train/tile_$IMAGE_SIZE

   echo "Testing DCNN on test tiles"
   python2 test_class_tiles.py -i test -t $IMAGE_SIZE -n 100

   rm -rf test/tile_$IMAGE_SIZE

   cls

done

####======================================================
#IMAGE_SIZE=96



####======================================================
#IMAGE_SIZE=128

##python2 retile.py -i train -t $IMAGE_SIZE -a .8

##python2 retrain.py --image_dir train/tile_$IMAGE_SIZE \
##    --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v2_100_$IMAGE_SIZE/classification/1 \
##    --how_many_training_steps 1000 \
##    --output_labels labels.txt --output_graph mobilenetv2_"${IMAGE_SIZE}".pb

##rm -rf /tmp/bottleneck
##rm -rf /tmp/checkpoint

#rm -rf train/tile_$IMAGE_SIZE

##python2 retile.py -i test -t $IMAGE_SIZE -a .8

##python2 test_class_tiles.py -i test -t $IMAGE_SIZE -n 100

#rm -rf test/tile_$IMAGE_SIZE

####======================================================
#IMAGE_SIZE=160

##python2 retile.py -i train -t $IMAGE_SIZE -a .7

##python2 retrain.py --image_dir train/tile_$IMAGE_SIZE \
##    --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v2_100_$IMAGE_SIZE/classification/1 \
##    --how_many_training_steps 1000 \
##    --output_labels labels.txt --output_graph mobilenetv2_"${IMAGE_SIZE}".pb

##rm -rf /tmp/bottleneck
##rm -rf /tmp/checkpoint

#rm -rf train/tile_$IMAGE_SIZE

##python2 retile.py -i test -t $IMAGE_SIZE -a .75

##python2 test_class_tiles.py -i test -t $IMAGE_SIZE -n 100

#rm -rf test/tile_$IMAGE_SIZE

####======================================================
#IMAGE_SIZE=192

##python2 retile.py -i train -t $IMAGE_SIZE -a .6

##python2 retrain.py --image_dir train/tile_$IMAGE_SIZE \
##    --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v2_100_$IMAGE_SIZE/classification/1 \
##    --how_many_training_steps 1000 \
##    --output_labels labels.txt --output_graph mobilenetv2_"${IMAGE_SIZE}".pb

##rm -rf /tmp/bottleneck
##rm -rf /tmp/checkpoint

#rm -rf train/tile_$IMAGE_SIZE

##python2 retile.py -i test -t $IMAGE_SIZE -a .7

##python2 test_class_tiles.py -i test -t $IMAGE_SIZE -n 100

#rm -rf test/tile_$IMAGE_SIZE

####======================================================
#IMAGE_SIZE=224

###python2 retile.py -i train -t $IMAGE_SIZE -a .5

##python2 retrain.py --image_dir train/tile_$IMAGE_SIZE \
##    --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v2_100_$IMAGE_SIZE/classification/1 \
##    --how_many_training_steps 1000 \
##    --output_labels labels.txt --output_graph mobilenetv2_"${IMAGE_SIZE}"_graph.pb

##rm -rf /tmp/bottleneck
##rm -rf /tmp/checkpoint

#rm -rf train/tile_$IMAGE_SIZE

##python2 retile.py -i test -t $IMAGE_SIZE -a .65

##python2 test_class_tiles.py -i test -t $IMAGE_SIZE -n 100

#rm -rf test/tile_$IMAGE_SIZE


