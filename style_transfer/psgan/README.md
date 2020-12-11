# PSGAN
Pose and Expression Robust Spatial-Aware GAN for Customizable Makeup Transfer
- Source: [https://github.com/wtjiang98/PSGAN](https://github.com/wtjiang98/PSGAN)

## Run example.
### Image
```buildoutcfg
# With an example image from the source repositry.
python3 psgan.py --onnx
```

### Video
```buildoutcfg
# With web camera on your PC.
# "0" is normally the device number of a built-in web camera.
python3 psgan.py -v 0 --onnx
```