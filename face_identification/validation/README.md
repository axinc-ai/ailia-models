# Validate ArcFace and VGGFace2

### usage

Compute a confusion matrix from a set of face images.

```bash
$ python3 validation.py -i ./CASIA-WebFace
```

The dataset folder must be divided into folders for each person.

An example of the folder structure is shown below.

```
./CASIA-WebFace/0000045/001.jpg
./CASIA-WebFace/0000045/002.jpg
./CASIA-WebFace/0000099/001.jpg
```

Cut out a facial region from the dataset.

```bash
$ python3 validation_dataset_gen.py -i ./dataset/no_crop -o ./dataset/crop
```
