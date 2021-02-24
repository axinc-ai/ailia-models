export OPTION=-b
cd ../
cd action_recognition/mars; python3 mars.py ${OPTION}
cd ../../action_recognition/st-gcn; python3 st-gcn.py ${OPTION}
cd ../../audio_processing/crnn_audio_classification; python3 crnn_audio_classification.py ${OPTION}
cd ../../audio_processing/deepspeech2; python3 deepspeech2.py ${OPTION}
cd ../../audio_processing/pytorch-dc-tts/; python3 pytorch-dc-tts.py ${OPTION}
cd ../../audio_processing/unet_source_separation/; python3 unet_source_separation.py ${OPTION}
cd ../../crowd_counting/crowdcount-cascaded-mtl; python3 crowdcount-cascaded-mtl.py ${OPTION}
cd ../../crowd_counting/c-3-framework; python3 c-3-framework.py ${OPTION}
cd ../../deep_fashion/clothing-detection; python3 clothing-detection.py ${OPTION}
cd ../../deep_fashion/fashionai-key-points-detection; python3 fashionai-key-points-detection.py -c blouse ${OPTION}
cd ../../deep_fashion/fashionai-key-points-detection; python3 fashionai-key-points-detection.py -c dress ${OPTION}
cd ../../deep_fashion/fashionai-key-points-detection; python3 fashionai-key-points-detection.py -c outwear ${OPTION}
cd ../../deep_fashion/fashionai-key-points-detection; python3 fashionai-key-points-detection.py -c skirt ${OPTION}
cd ../../deep_fashion/fashionai-key-points-detection; python3 fashionai-key-points-detection.py -c trousers ${OPTION}
cd ../../deep_fashion/mmfashion; python3 mmfashion.py ${OPTION}
cd ../../depth_estimation/midas; python3 midas.py ${OPTION}
cd ../../depth_estimation/monodepth2; python3 monodepth2.py ${OPTION}
cd ../../depth_estimation/fcrn-depthprediction; python3 fcrn-depthprediction.py ${OPTION}
cd ../../face_detection/blazeface; python3 blazeface.py ${OPTION}
cd ../../face_detection/dbface; python3 dbface.py ${OPTION}
cd ../../face_detection/face-mask-detection; python3 face-mask-detection.py ${OPTION}
cd ../../face_detection/face-mask-detection; python3 face-mask-detection.py -a mb2-ssd ${OPTION}
cd ../../face_detection/yolov1-face; python3 yolov1-face.py ${OPTION}
cd ../../face_detection/yolov3-face; python3 yolov3-face.py ${OPTION}
cd ../../face_identification/arcface; python3 arcface.py ${OPTION}
cd ../../face_identification/insightface; python3 insightface.py ${OPTION}
cd ../../face_identification/vggface2; python3 vggface2.py ${OPTION}
cd ../../face_recognition/face_alignment; python3 face_alignment.py ${OPTION}
cd ../../face_recognition/face_alignment; python3 face_alignment.py --active_3d ${OPTION}
cd ../../face_recognition/face_classification; python3 face_classification.py ${OPTION}
cd ../../face_recognition/facial_feature; python3 facial_feature.py ${OPTION}
cd ../../face_recognition/gazeml; python3 gazeml.py ${OPTION}
cd ../../face_recognition/prnet; python3 prnet.py ${OPTION}
cd ../../face_recognition/facemesh; python3 facemesh.py ${OPTION}
cd ../../face_recognition/mediapipe_iris; python3 mediapipe_iris.py ${OPTION}
cd ../../generative_adversarial_networks/council-GAN; python3 council-gan.py --glasses ${OPTION}
cd ../../generative_adversarial_networks/council-GAN; python3 council-gan.py --m2f ${OPTION}
cd ../../generative_adversarial_networks/council-GAN; python3 council-gan.py --anime ${OPTION}
cd ../../generative_adversarial_networks/pytorch-gan; python3 pytorch-gnet.py -m celeb ${OPTION}
cd ../../generative_adversarial_networks/pytorch-gan; python3 pytorch-gnet.py -m anime ${OPTION}
cd ../../hand_detection/yolov3-hand; python3 yolov3-hand.py ${OPTION}
cd ../../hand_detection/hand_detection_pytorch python3 hand_detection_pytorch.py ${OPTION}
cd ../../hand_detection/blazepalm; python3 blazepalm.py ${OPTION}
cd ../../hand_recognition/blazehand; python3 blazehand.py ${OPTION}
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
cd ../../image_manipulation/u2net_portrait; python3 u2net_portrait.py ${OPTION}
cd ../../image_manipulation/style2paints; python3 style2paints.py ${OPTION}
cd ../../image_segmentation/deep-image-matting; python3 deep-image-matting.py ${OPTION}
cd ../../image_segmentation/deeplabv3; python3 deeplabv3.py ${OPTION}
cd ../../image_segmentation/hair_segmentation; python3 hair_segmentation.py ${OPTION}
cd ../../image_segmentation/hrnet_segmentation; python3 hrnet_segmentation.py ${OPTION}
cd ../../image_segmentation/pspnet-hair-segmentation; python3 pspnet-hair-segmentation.py ${OPTION}
cd ../../image_segmentation/u2net; python3 u2net.py ${OPTION}
cd ../../image_segmentation/u2net; python3 u2net.py -a small ${OPTION}
cd ../../image_segmentation/human_part_segmentation; python3 human_part_segmentation.py ${OPTION}
cd ../../image_segmentation/pytorch-unet; python3 pytorch-unet.py ${OPTION}
cd ../../image_segmentation/semantic-segmentation-mobilenet-v3; python3 semantic-segmentation-mobilenet-v3.py ${OPTION}
cd ../../image_segmentation/yet-another-anime-segmenter python3 yet-another-anime-segmenter.py ${OPTION}
cd ../../neural_language_processing/bert; python3 bert.py ${OPTION}
cd ../../neural_language_processing/bert_tweets_sentiment; python3 bert_tweets_sentiment.py ${OPTION}
cd ../../neural_language_processing/bert_maskedlm; python3 bert_maskedlm.py ${OPTION}
cd ../../neural_language_processing/bert_ner; python3 bert_ner.py ${OPTION}
cd ../../neural_language_processing/bert_question_answering; python3 bert_question_answering.py ${OPTION}
cd ../../neural_language_processing/bert_sentiment_analysis; python3 bert_sentiment_analysis.py ${OPTION}
cd ../../neural_language_processing/bert_tweets_sentiment; python3 bert_tweets_sentiment.py ${OPTION}
cd ../../neural_language_processing/bert_zero_shot_classification; python3 bert_zero_shot_classification.py ${OPTION}
cd ../../object_detection/centernet; python3 centernet.py ${OPTION}
cd ../../object_detection/m2det; python3 m2det.py ${OPTION}
cd ../../object_detection/maskrcnn; python3 maskrcnn.py ${OPTION}
cd ../../object_detection/mobilenet_ssd; python3 mobilenet_ssd.py ${OPTION}
cd ../../object_detection/yolov1-tiny; python3 yolov1-tiny.py ${OPTION}
cd ../../object_detection/yolov2; python3 yolov2.py ${OPTION}
cd ../../object_detection/yolov3; python3 yolov3.py ${OPTION}
cd ../../object_detection/yolov3-tiny; python3 yolov3-tiny.py ${OPTION}
cd ../../object_detection/yolov4; python3 yolov4.py ${OPTION}
cd ../../object_detection/yolov4-tiny; python3 yolov4-tiny.py ${OPTION}
cd ../../object_detection/yolov5; python3 yolov5.py ${OPTION}
cd ../../object_detection/pedestrian_detection; python3 pedestrian_detection.py ${OPTION}
cd ../../object_detection/efficientdet; python3 efficientdet.py ${OPTION}
cd ../../object_tracking/deepsort; python3 deepsort.py ${OPTION}
cd ../../object_tracking/person_reid_baseline_pytorch; python3 person_reid_baseline_pytorch.py ${OPTION}
cd ../../point_segmentation/pointnet_pytorch python3 pointnet_pytorch.py ${OPTION}
cd ../../pose_estimation/3d-pose-baseline; python3 3d-pose-baseline.py ${OPTION}
cd ../../pose_estimation/lightweight-human-pose-estimation; python3 lightweight-human-pose-estimation.py ${OPTION}
cd ../../pose_estimation/lightweight-human-pose-estimation-3d; python3 lightweight-human-pose-estimation-3d.py ${OPTION}
cd ../../pose_estimation/openpose; python3 openpose.py ${OPTION}
cd ../../pose_estimation/pose_resnet; python3 pose_resnet.py ${OPTION}
cd ../../pose_estimation/blazepose; python3 blazepose.py ${OPTION}
cd ../../pose_estimation/3dmppe_posenet; python3 3dmppe_posenet.py ${OPTION}
cd ../../pose_estimation/efficientpose; python3 efficientpose.py ${OPTION}
cd ../../rotation_prediction/rotnet; python3 rotnet.py ${OPTION}
cd ../../style_transfer/adain; python3 adain.py ${OPTION}
cd ../../super_resolution/srresnet; python3 srresnet.py ${OPTION}
cd ../../text_detection/craft_pytorch; python3 craft_pytorch.py ${OPTION}
cd ../../text_detection/pixel_link; python3 pixel_link.py ${OPTION}
cd ../../text_detection/east; python3 east.py ${OPTION}
cd ../../text_recognition/etl; python3 etl.py ${OPTION}
cd ../../text_recognition/deep-text-recognition-benchmark; python3 deep-text-recognition-benchmark.py ${OPTION}
cd ../../text_recognition/crnn.pytorch; python3 crnn.pytorch.py ${OPTION}
#cd ../../commercial_model/acculus-pose; python3 acculus-hand.py ${OPTION}
#cd ../../commercial_model/acculus-pose; python3 acculus-pose.py ${OPTION}
#cd ../../commercial_model/acculus-pose; python3 acculus-up-pose.py ${OPTION}
