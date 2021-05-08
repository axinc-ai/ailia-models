from fvcore.common.config import CfgNode

"""
This file defines default options of configurations.
It will be further merged by yaml files and options from
the command-line.
Note that *any* hyper-parameters should be firstly defined
here to enable yaml and command-line configuration.
"""

_C = CfgNode()


# Paths for logging and saving
_C.LOG = CfgNode()
_C.LOG.LOG_PATH = 'log/'
_C.LOG.SNAPSHOT_PATH = 'snapshot/'
_C.LOG.VIS_PATH = 'visulization/'
_C.LOG.SNAPSHOT_STEP = 1024
_C.LOG.LOG_STEP = 8
_C.LOG.VIS_STEP = 2048

# Data settings
_C.DATA = CfgNode()
_C.DATA.PATH = './data'
_C.DATA.NUM_WORKERS = 4
_C.DATA.BATCH_SIZE = 1
_C.DATA.IMG_SIZE = 256

# Training hyper-parameters
_C.TRAINING = CfgNode()
_C.TRAINING.G_LR = 2e-4
_C.TRAINING.D_LR = 2e-4
_C.TRAINING.BETA1 = 0.5
_C.TRAINING.BETA2 = 0.999
_C.TRAINING.C_DIM = 2
_C.TRAINING.G_STEP = 1
_C.TRAINING.NUM_EPOCHS = 50
_C.TRAINING.NUM_EPOCHS_DECAY = 0

# Loss weights
_C.LOSS = CfgNode()
_C.LOSS.LAMBDA_A = 10.0
_C.LOSS.LAMBDA_B = 10.0
_C.LOSS.LAMBDA_IDT = 0.5
_C.LOSS.LAMBDA_CLS = 1
_C.LOSS.LAMBDA_REC = 10
_C.LOSS.LAMBDA_HIS = 1
_C.LOSS.LAMBDA_SKIN = 0.1
_C.LOSS.LAMBDA_EYE = 1
_C.LOSS.LAMBDA_HIS_LIP = _C.LOSS.LAMBDA_HIS
_C.LOSS.LAMBDA_HIS_SKIN = _C.LOSS.LAMBDA_HIS * _C.LOSS.LAMBDA_SKIN
_C.LOSS.LAMBDA_HIS_EYE = _C.LOSS.LAMBDA_HIS * _C.LOSS.LAMBDA_EYE
_C.LOSS.LAMBDA_VGG = 5e-3

# Model structure
_C.MODEL = CfgNode()
_C.MODEL.G_CONV_DIM = 64
_C.MODEL.D_CONV_DIM = 64
_C.MODEL.G_REPEAT_NUM = 6
_C.MODEL.D_REPEAT_NUM = 3
_C.MODEL.NORM = "SN"
_C.MODEL.WEIGHTS = "assets/models"


# Preprocessing
_C.PREPROCESS = CfgNode()
_C.PREPROCESS.UP_RATIO = 0.6 / 0.85  # delta_size / face_size
_C.PREPROCESS.DOWN_RATIO = 0.2 / 0.85  # delta_size / face_size
_C.PREPROCESS.WIDTH_RATIO = 0.2 / 0.85  # delta_size / face_size
_C.PREPROCESS.LIP_CLASS = [7, 9]
_C.PREPROCESS.FACE_CLASS = [1, 6]
_C.PREPROCESS.LANDMARK_POINTS = 68

# Postprocessing
_C.POSTPROCESS = CfgNode()
_C.POSTPROCESS.WILL_DENOISE = False


def get_config()->CfgNode:
    return _C
