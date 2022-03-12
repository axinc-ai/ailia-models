#!/bin/bash

cate=$1

rm -f train/*.png
rm -f images/*.png
rm -f gt_masks/*.png

find mvtec_anomaly_detection/$cate/train/ -name "*.png" -exec bash -c 'echo "cp {} train/`basename {}`"; cp {} train/`basename {}`' \;

if [ $? != 0 ]; then
  exit -1
fi

for p in $( ls mvtec_anomaly_detection/$cate/test ); do   
  find mvtec_anomaly_detection/$cate/test/$p -name "*.png" -exec bash -c 'echo "cp {} images/$0_`basename {}`"; cp {} images/$0_`basename {}`' $p \;
  find mvtec_anomaly_detection/$cate/ground_truth/$p -name "*.png" -exec bash -c 'echo "cp {} gt_masks/$0_`basename {}`"; cp {} gt_masks/$0_`basename {}`' $p \; 2>/dev/null
done
