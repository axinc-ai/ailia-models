import numpy as np
from .opts import opts

opt_str = 'ddd --demo ../images/17790319373_bd19b24cfc_k.jpg --load_model ../models/ddd_dlav0.pth --arch dlav0_34'.split()
opt = opts().init(opt_str)

MEAN = np.asarray([[[0.485, 0.456, 0.406]]])
STD = np.asarray([[[0.229, 0.224, 0.225]]])
CALIB = np.asarray([[ 7.070493e+02,  0.000000e+00,  6.040814e+02,  4.575831e+01],
       [ 0.000000e+00,  7.070493e+02,  1.805066e+02, -3.454157e-01],
       [ 0.000000e+00,  0.000000e+00,  1.000000e+00,  4.981016e-03]])