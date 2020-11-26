export OPTION=-b
cd ../
cd action_recognition/mars; python3 mars.py ${OPTION}
cd ../../crowd_counting/crowdcount-cascaded-mtl; python3 crowdcount.py ${OPTION}
cd ../../deep_fashion/clothing-detection; python3 clothing-detection.py ${OPTION}
cd ../../depth_estimation/midas; python3 midas.py ${OPTION}
cd ../../depth_estimation/monodepth2; python3 monodepth2.py ${OPTION}
cd ../../face_detection/blazeface; python3 blazeface.py ${OPTION}
cd ../../face_detection/dbface; python3 dbface.py ${OPTION}
cd ../../face_detection/face-mask-detection; python3 face-mask-detection.py ${OPTION}
cd ../../face_detection/face-mask-detection; python3 face-mask-detection.py -a mb2-ssd ${OPTION}
cd ../../face_detection/yolov1-face; python3 yolov1-face.py ${OPTION}
cd ../../face_detection/yolov3-face; python3 yolov3-face.py ${OPTION}
cd ../../face_identification/arcface; python3 arcface.py ${OPTION}
cd ../../face_identification/vggface2; python3 vggface2.py ${OPTION}
cd ../../face_recognition/face_alignment; python3 face_alignment.py ${OPTION}
cd ../../face_recognition/face_alignment; python3 face_alignment.py --active_3d ${OPTION}
cd ../../face_recognition/face_classification; python3 face_classification.py ${OPTION}
cd ../../face_recognition/facial_feature; python3 facial_feature.py ${OPTION}
cd ../../face_recognition/gazeml; python3 gazeml.py ${OPTION}
cd ../../face_recognition/prnet; python3 prnet.py ${OPTION}
cd ../../generative_adversarial_networks/council-GAN; python3 council-gan.py --glasses ${OPTION}
cd ../../generative_adversarial_networks/council-GAN; python3 council-gan.py --m2f ${OPTION}
cd ../../generative_adversarial_networks/council-GAN; python3 council-gan.py --anime ${OPTION}
cd ../../generative_adversarial_networks/pytorch-gan; python3 pytorch-gnet.py -m celeb ${OPTION}
cd ../../generative_adversarial_networks/pytorch-gan; python3 pytorch-gnet.py -m anime ${OPTION}
cd ../../hand_detection/yolov3-hand; python3 yolov3-hand.py ${OPTION}
cd ../../hand_detection/hand_detection_pytorch python3 hand_detection_pytorch.py ${OPTION}
cd ../../image_captioning/illustration2vec; python3 illustration2vec.py ${OPTION}
cd ../../image_captioning/image_captioning_pytorch; python3 image_captioning_pytorch.py ${OPTION}
cd ../../image_classification/efficientnet; python3 efficientnet.py ${OPTION}
cd ../../image_classification/googlenet; python3 googlenet.py ${OPTION}
cd ../../image_classification/inceptionv3; python3 inceptionv3.py ${OPTION}
cd ../../image_classification/mobilenetv2; python3 mobilenetv2.py ${OPTION}
cd ../../image_classification/mobilenetv3; python3 mobilenetv3.py ${OPTION}
cd ../../image_classification/partialconv; python3 partialconv.py ${OPTION}
cd ../../image_classification/resnet50; python3 resnet50.py ${OPTION}
cd ../../image_classification/vgg16; python3 vgg16.py ${OPTION}
cd ../../image_manipulation/dewarpnet; python3 dewarpnet.py ${OPTION}
cd ../../image_manipulation/illnet; python3 illnet.py ${OPTION}
cd ../../image_manipulation/noise2noise; python3 noise2noise.py ${OPTION}
cd ../../image_manipulation/colorization; python3 colorization.py ${OPTION}
cd ../../image_segmentation/deep-image-matting; python3 deep-image-matting.py ${OPTION}
cd ../../image_segmentation/deeplabv3; python3 deeplabv3.py ${OPTION}
cd ../../image_segmentation/hair_segmentation; python3 hair_segmentation.py ${OPTION}
cd ../../image_segmentation/hrnet_segmentation; python3 hrnet_segmentation.py ${OPTION}
cd ../../image_segmentation/pspnet-hair-segmentation; python3 pspnet-hair-segmentation.py ${OPTION}
cd ../../image_segmentation/u2net; python3 u2net.py ${OPTION}
cd ../../image_segmentation/u2net; python3 u2net.py -a small ${OPTION}
cd ../../image_segmentation/human_part_segmentation; python3 human_part_segmentation.py ${OPTION}
cd ../../neural_language_processing/bert; python3 bert.py ${OPTION}
cd ../../object_detection/centernet; python3 centernet.py ${OPTION}
cd ../../object_detection/m2det; python3 m2det.py ${OPTION}
cd ../../object_detection/maskrcnn; python3 maskrcnn.py ${OPTION}
cd ../../object_detection/mobilenet_ssd; python3 mobilenet_ssd.py ${OPTION}
cd ../../object_detection/yolov1-tiny; python3 yolov1-tiny.py ${OPTION}
cd ../../object_detection/yolov2; python3 yolov2.py ${OPTION}
cd ../../object_detection/yolov3; python3 yolov3.py ${OPTION}
cd ../../object_detection/yolov3-tiny; python3 yolov3-tiny.py ${OPTION}
cd ../../object_detection/yolov4; python3 yolov4.py ${OPTION}
cd ../../object_tracking/deepsort; python3 deepsort.py ${OPTION}
cd ../../pose_estimation/3d-pose-baseline; python3 3d-pose-baseline.py ${OPTION}
cd ../../pose_estimation/lightweight-human-pose-estimation; python3 lightweight-human-pose-estimation.py ${OPTION}
cd ../../pose_estimation/lightweight-human-pose-estimation-3d; python3 lightweight-human-pose-estimation-3d.py ${OPTION}
cd ../../pose_estimation/openpose; python3 openpose.py ${OPTION}
cd ../../pose_estimation/pose_resnet; python3 pose_resnet.py ${OPTION}
cd ../../rotation_prediction/rotnet; python3 rotnet.py ${OPTION}
cd ../../style_transfer/adain; python3 adain.py ${OPTION}
cd ../../super_resolution/srresnet; python3 srresnet.py ${OPTION}
cd ../../text_recognition/etl; python3 etl.py ${OPTION}
#cd ../../commercial_model/acculus-pose; python3 acculus-hand.py ${OPTION}
#cd ../../commercial_model/acculus-pose; python3 acculus-pose.py ${OPTION}
#cd ../../commercial_model/acculus-pose; python3 acculus-up-pose.py ${OPTION}
