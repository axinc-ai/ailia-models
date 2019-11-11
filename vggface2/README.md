# VGGFace2 Dataset for Face Recognition

## Input

- image_a : ![Input](couple_a.jpg)
- image_b : ![Input](couple_b.jpg)
- image_c : ![Input](couple_c.jpg)

Shape : (1, 3, 224, 224)
Range : [-128.0, 127.0]

## Output

Feature Shape : (1,1,1,2048)

```
image_a vs image_b =  1.143173
not same person
image_a vs image_c =  0.7116655
same person
```

## Reference

- [VGGFace2](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)
- [VGGFace2 Dataset for Face Recognition](https://github.com/ox-vgg/vgg_face2)

## Framework

Caffe

## Model Format

CaffeModel