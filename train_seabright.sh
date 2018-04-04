##https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0

python2 retrain.py --image_dir seabright/autoclassified_gc96 \
    --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/classification/1 \
    --how_many_training_steps 1000 \
    --output_labels seabright/seabright_labels.txt --output_graph seabright/seabright_mobilenetv2_96_graph_orig.pb

rm -rf /tmp/bottleneck
rm -rf /tmp/checkpoint


#prefix="seabright"

#IMAGE_SIZE=128 ##128,160,192, or 224px
###size of the model as a fraction of the largest MobileNet: 1.0, 0.75, 0.50, or 0.25
#ARCHITECTURE="mobilenet_1.0_${IMAGE_SIZE}" 

##======================================================
##==== ORIG
#python2 retrain.py --bottleneck_dir=../$prefix/bottlenecks --how_many_training_steps=1000 --model_dir=models/ --summaries_dir=../$prefix/training_summaries/"${ARCHITECTURE}" --output_graph=../$prefix/orig_graph_"${ARCHITECTURE}".pb --output_labels=../$prefix/labels_orig.txt --architecture="${ARCHITECTURE}" --image_dir=../$prefix/autoclassified128

#rm -rf ../$prefix/bottlenecks
#rm -rf ../$prefix/training_summaries

##======================================================
##==== TEX
#python2 retrain.py --bottleneck_dir=../$prefix/bottlenecks --how_many_training_steps=1000 --model_dir=models/ --summaries_dir=../$prefix/training_summaries/"${ARCHITECTURE}" --output_graph=../$prefix/texaug_graph_"${ARCHITECTURE}".pb --output_labels=../$prefix/labels_tex.txt --architecture="${ARCHITECTURE}" --image_dir=../$prefix/autoclassified_texaug128

#rm -rf ../$prefix/bottlenecks
#rm -rf ../$prefix/training_summaries

##======================================================
##==== MERGED (ALL)
#python2 retrain.py --bottleneck_dir=../$prefix/bottlenecks --how_many_training_steps=1000 --model_dir=models/ --summaries_dir=../$prefix/training_summaries/"${ARCHITECTURE}" --output_graph=../$prefix/merged_graph_"${ARCHITECTURE}".pb --output_labels=../$prefix/labels_merged.txt --architecture="${ARCHITECTURE}" --image_dir=../$prefix/autoclassified_merged128

#rm -rf ../$prefix/bottlenecks
#rm -rf ../$prefix/training_summaries

##======================================================
##==== MERGED (JUST NATURAL)
#python2 retrain.py --bottleneck_dir=../$prefix/bottlenecks --how_many_training_steps=1000 --model_dir=models/ --summaries_dir=../$prefix/training_summaries/"${ARCHITECTURE}" --output_graph=../$prefix/merged_nat_graph_"${ARCHITECTURE}".pb --output_labels=../$prefix/labels_merged_nat.txt --architecture="${ARCHITECTURE}" --image_dir=../$prefix/autoclassified_merged128_nat

#rm -rf ../$prefix/bottlenecks
#rm -rf ../$prefix/training_summaries


#======================================================
#==== MERGED (JUST NATURAL)
#python2 retrain.py --bottleneck_dir=../$prefix/bottlenecks --how_many_training_steps=1000 --model_dir=models/ --summaries_dir=../$prefix/training_summaries/"${ARCHITECTURE}" --output_graph=../$prefix/best_graph_"${ARCHITECTURE}".pb --output_labels=../$prefix/labels_best.txt --architecture="${ARCHITECTURE}" --image_dir=../$prefix/autoclassified_best

#rm -rf ../$prefix/bottlenecks
#rm -rf ../$prefix/training_summaries

