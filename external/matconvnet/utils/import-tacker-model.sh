#! /bin/bash
# brief: Import various CNN models from the web
# author: Andrea Vedaldi

# Models are written to <MATCONVNET>/data/models
# You can delete <MATCONVNET>/data/tmp after conversion
converter="python import-caffe.py"
out=./tracker.mat
test ! -e "$out" && \
$converter \
--caffe-variant=caffe \
 --preproc=caffe \
--average-image="./imagenet_mean.binaryproto" \
"/opt/CF2/GOTURN-master/nets/tracker_n.prototxt" \
"/opt/CF2/CF2-master/matconvnet-model/tracker.caffemodel" \
"$out"

