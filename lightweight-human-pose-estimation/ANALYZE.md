# lightweight-human-pose-estimation

## Usage

Display heatmap for analyze

```
python3 lightweight-human-pose-estimation-analyze.py
```

## Input

![Input](balloon.png)

Shape : (1, 3, 240, 320)
Range : [-0.5, 0.5]

## Output

![Confidence](confidence.png)

- Paf : (1, 38,  30, 40)
- Range : [0, 1.0]

![Paf](paf.png)

## Keypoint Order

[Pose Output Format (COCO)](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md)

```
COCO_KEYPOINT_NOSE = (0)
COCO_KEYPOINT_NECK = (1)
COCO_KEYPOINT_SHOULDER_RIGHT = (2)
COCO_KEYPOINT_ELBOW_RIGHT = (3)
COCO_KEYPOINT_WRIST_RIGHT = (4)
COCO_KEYPOINT_SHOULDER_LEFT = (5)
COCO_KEYPOINT_ELBOW_LEFT = (6)
COCO_KEYPOINT_WRIST_LEFT = (7)
COCO_KEYPOINT_HIP_RIGHT = (8)
COCO_KEYPOINT_KNEE_RIGHT = (9)
COCO_KEYPOINT_ANKLE_RIGHT = (10)
COCO_KEYPOINT_HIP_LEFT = (11)
COCO_KEYPOINT_KNEE_LEFT = (12)
COCO_KEYPOINT_ANKLE_LEFT = (13)
COCO_KEYPOINT_EYE_RIGHT = (14)
COCO_KEYPOINT_EYE_LEFT = (15)
COCO_KEYPOINT_EAR_RIGHT = (16)
COCO_KEYPOINT_EAR_LEFT = (17)
COCO_KEYPOINT_BACKGROUND = (18)
```
