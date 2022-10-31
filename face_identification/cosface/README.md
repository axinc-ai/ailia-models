# CosFace

### Paper
 - CosFace: Large Margin Cosine Loss for Deep Face Recognition
   - https://arxiv.org/abs/1801.09414

### Reference
 - https://github.com/MuggleWang/CosFace_pytorch

### Sample Code
  - cosface.py  # Output similarity from two face images
  - requirements.txt # Dependencies
#### Usages
 - Install dependencies
   - `pip install -r requirements.txt`
 - Show Help 
   - `python cosface.py --help`
 - Get similarity from 2 face images
   - `python cosface.py --target_image=image_target.jpg --id_image=image_id.jpg --onnx=cosface_sphere20.onnx`

#### Input
- Ailia input shape : (1, 3, 96, 112)

| Sample input file | Image                       |
|-------------------|-----------------------------|
 | image_target.jpg  | ![Input](image_target.jpg)] |
 | image_id.jpg      | ![Input](image_id.jpg)      |
