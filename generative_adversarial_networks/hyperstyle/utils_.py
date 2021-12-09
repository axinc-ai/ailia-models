import os
import cv2 

# ======================
# PARAMETERS
# ======================
AILIA_MODELS_BASE_DIR = '../..'

CHECKPOINTS = 'checkpoints/'
AVERAGE = 'average/'
INTERFACEGAN_EDITING = 'editing/interfacegan_directions'

CHOICES = ["cartoon", "disney-princess", "sketch", "pixar"]

IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 1024

RESIZE_HEIGHT = 256
RESIZE_WIDTH = 256

IMAGE_PATH = 'img/watson.jpg'
SAVE_IMAGE_PATH = 'img/output.png'
ALIGNED_PATH = 'img/aligned/'

# ======================
# FUNCTIONS
# ======================
def np2im(var, input=False):
    var = var.astype('float32')
    var = cv2.cvtColor(
        var.transpose(1, 2, 0),
        cv2.COLOR_RGB2BGR
    )
    if not input:
        var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return var.astype('uint8')