from enum import Enum
import numpy as np
import cv2
import urllib
from dataclasses import dataclass

class ModelType(Enum):
	eth3d = 0
	middlebury = 1
	flyingthings = 2

@dataclass
class CameraConfig:
    baseline: float
    f: float

def load_img(url):
	req = urllib.request.urlopen(url)
	arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
	return cv2.imdecode(arr, -1) # 'Load it as it is'

def draw_disparity(disparity_map):

	disparity_map = disparity_map.astype(np.uint8)
	norm_disparity_map = (255*((disparity_map-np.min(disparity_map))/(np.max(disparity_map) - np.min(disparity_map))))
	return cv2.applyColorMap(cv2.convertScaleAbs(norm_disparity_map,1), cv2.COLORMAP_MAGMA)

def draw_depth(depth_map, max_dist):
	
	norm_depth_map = 255*(1-depth_map/max_dist)
	norm_depth_map[norm_depth_map < 0] =0
	norm_depth_map[depth_map == 0] =0

	return cv2.applyColorMap(cv2.convertScaleAbs(norm_depth_map,1), cv2.COLORMAP_MAGMA)
