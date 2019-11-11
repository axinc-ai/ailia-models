# Real-time face detection and emotion/gender classification

## Input

![Input](lenna.png)

Shape : (1, 1, 64, 64)
Range : [-1.0, 1.0]

## Output

```
emotion_class_count=3
+ idx=0
  category=6[ neutral ]
  prob=0.411855548620224
+ idx=1
  category=4[ sad ]
  prob=0.1994263231754303
+ idx=2
  category=0[ angry ]
  prob=0.19452838599681854

gender_class_count=2
+ idx=0
  category=0[ female ]
  prob=0.8007728457450867
+ idx=1
  category=1[ male ]
  prob=0.19922710955142975
```

## Reference

[Real-time face detection and emotion/gender classification](https://github.com/oarriaga/face_classification)

## Framework

Keras

## Model Format

CaffeModel