#!/bin/bash

# Downloading models for pose estimation
OPENPOSE_URL="http://posefs1.perception.cs.cmu.edu/OpenPose/models/"

# Body (COCO)
COCO_FOLDER="pose/coco/"
OUT_FOLDER="${COCO_FOLDER}"
COCO_MODEL=${COCO_FOLDER}"pose_iter_440000.caffemodel"
# wget -c ${OPENPOSE_URL}${COCO_MODEL} -P ${OUT_FOLDER}
curl ${OPENPOSE_URL}${COCO_MODEL} -o ${OUT_FOLDER}/"pose_iter_440000.caffemodel"
