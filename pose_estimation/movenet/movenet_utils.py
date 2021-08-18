import numpy as np
import cv2

# preprocessing for input image to movenet model
def crop_and_padding(image,input_size):

  h,w = image.shape[0],image.shape[1]

  if h > w:
    rh = input_size
    rw = int(input_size * w / h)
    input_image = cv2.resize(image, (rw,rh), interpolation = cv2.INTER_AREA)
    pad = int((input_size-rw) / 2)
    input_image = cv2.copyMakeBorder(input_image, 0, 0, pad, pad, cv2.BORDER_CONSTANT, (0,0,0))
    input_image = cv2.resize(input_image, (input_size,input_size), interpolation = cv2.INTER_AREA)
  else:
    rw = input_size
    rh = int(input_size * h / w)
    input_image = cv2.resize(image, (rw,rh), interpolation = cv2.INTER_AREA)
    pad = int((input_size-rh) / 2)
    input_image = cv2.copyMakeBorder(input_image, pad, pad, 0, 0, cv2.BORDER_CONSTANT, (0,0,0))
    input_image = cv2.resize(input_image, (input_size,input_size), interpolation = cv2.INTER_AREA)

  return input_image, pad/input_size

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Maps bones to a matplotlib color name.
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): (179,30,179), # m
    (0, 2): (12,187,192), # c
    (1, 3): (179,30,179), # m
    (2, 4): (12,187,192), # c
    (0, 5): (179,30,179), # m
    (0, 6): (12,187,192), # c
    (5, 7): (179,30,179), # m
    (7, 9): (179,30,179), # m
    (6, 8): (12,187,192), # c
    (8, 10): (12,187,192), # c
    (5, 6): (191,191,51), # y
    (5, 11): (179,30,179), # m
    (6, 12): (12,187,192), # c
    (11, 12): (191,191,51), # y
    (11, 13): (179,30,179), # m
    (13, 15): (179,30,179), # m
    (12, 14): (12,187,192), # c
    (14, 16): (12,187,192) # c
}

keypoint_threshold=0.11

def draw_prediction_on_image(image, keypoints_with_scores):

  height,width,_ = image.shape
  num_instances, _, _, _ = keypoints_with_scores.shape

  # set line width
  if width < 480:
    edge_line_width = 2
    circle_line_width = 3
  elif 480 <= width and width < 720:
    edge_line_width = 5
    circle_line_width = 7
  else:
    edge_line_width = 15
    circle_line_width = 20

  # calculate absolute xy
  for idx in range(num_instances):
    kpts_x = keypoints_with_scores[0, idx, :, 1]
    kpts_y = keypoints_with_scores[0, idx, :, 0]
    kpts_scores = keypoints_with_scores[0, idx, :, 2]
    kpts_absolute_xy = np.stack( [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)

  # draw edge
  for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
    if (kpts_scores[edge_pair[0]] > keypoint_threshold and kpts_scores[edge_pair[1]] > keypoint_threshold):
      x1 = kpts_absolute_xy[edge_pair[0], 0]
      y1 = kpts_absolute_xy[edge_pair[0], 1]
      x2 = kpts_absolute_xy[edge_pair[1], 0]
      y2 = kpts_absolute_xy[edge_pair[1], 1]
      cv2.line(image, (x1, y1), (x2, y2), color, edge_line_width)
  
  # draw key points
  for idx in range(0,len(kpts_absolute_xy)):
    kpt_xy = kpts_absolute_xy[idx]
    if kpts_scores[idx] > keypoint_threshold:
      cv2.circle(image, (kpt_xy[0], kpt_xy[1]), circle_line_width, color=(255,21,147), thickness=-1, lineType=cv2.LINE_8, shift=0)

  return image


MIN_CROP_KEYPOINT_SCORE = 0.2

def init_crop_region(image_height, image_width):
  """Defines the default crop region.

  The function provides the initial crop region (pads the full image from both
  sides to make it a square image) when the algorithm cannot reliably determine
  the crop region from the previous frame.
  """
  if image_width > image_height:
    box_height = image_width / image_height
    box_width = 1.0
    y_min = (image_height / 2 - image_width / 2) / image_height
    x_min = 0.0
  else:
    box_height = 1.0
    box_width = image_height / image_width
    y_min = 0.0
    x_min = (image_width / 2 - image_height / 2) / image_width

  return {
    'y_min': y_min,
    'x_min': x_min,
    'y_max': y_min + box_height,
    'x_max': x_min + box_width,
    'height': box_height,
    'width': box_width
  }

def torso_visible(keypoints):
  """Checks whether there are enough torso keypoints.

  This function checks whether the model is confident at predicting one of the
  shoulders/hips which is required to determine a good crop region.
  """
  return ((keypoints[0, 0, KEYPOINT_DICT['left_hip'], 2] >
           MIN_CROP_KEYPOINT_SCORE or
          keypoints[0, 0, KEYPOINT_DICT['right_hip'], 2] >
           MIN_CROP_KEYPOINT_SCORE) and
          (keypoints[0, 0, KEYPOINT_DICT['left_shoulder'], 2] >
           MIN_CROP_KEYPOINT_SCORE or
          keypoints[0, 0, KEYPOINT_DICT['right_shoulder'], 2] >
           MIN_CROP_KEYPOINT_SCORE))

def determine_torso_and_body_range(
    keypoints, target_keypoints, center_y, center_x):
  """Calculates the maximum distance from each keypoints to the center location.

  The function returns the maximum distances from the two sets of keypoints:
  full 17 keypoints and 4 torso keypoints. The returned information will be
  used to determine the crop size. See determineCropRegion for more detail.
  """
  torso_joints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
  max_torso_yrange = 0.0
  max_torso_xrange = 0.0
  for joint in torso_joints:
    dist_y = abs(center_y - target_keypoints[joint][0])
    dist_x = abs(center_x - target_keypoints[joint][1])
    if dist_y > max_torso_yrange:
      max_torso_yrange = dist_y
    if dist_x > max_torso_xrange:
      max_torso_xrange = dist_x

  max_body_yrange = 0.0
  max_body_xrange = 0.0
  for joint in KEYPOINT_DICT.keys():
    if keypoints[0, 0, KEYPOINT_DICT[joint], 2] < MIN_CROP_KEYPOINT_SCORE:
      continue
    dist_y = abs(center_y - target_keypoints[joint][0]);
    dist_x = abs(center_x - target_keypoints[joint][1]);
    if dist_y > max_body_yrange:
      max_body_yrange = dist_y

    if dist_x > max_body_xrange:
      max_body_xrange = dist_x

  return [max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange]

def determine_crop_region(
      keypoints, image_height,
      image_width):
  """Determines the region to crop the image for the model to run inference on.

  The algorithm uses the detected joints from the previous frame to estimate
  the square region that encloses the full body of the target person and
  centers at the midpoint of two hip joints. The crop size is determined by
  the distances between each joints and the center point.
  When the model is not confident with the four torso joint predictions, the
  function returns a default crop which is the full image padded to square.
  """
  target_keypoints = {}
  for joint in KEYPOINT_DICT.keys():
    target_keypoints[joint] = [
      keypoints[0, 0, KEYPOINT_DICT[joint], 0] * image_height,
      keypoints[0, 0, KEYPOINT_DICT[joint], 1] * image_width
    ]

  if torso_visible(keypoints):
    center_y = (target_keypoints['left_hip'][0] +
                target_keypoints['right_hip'][0]) / 2;
    center_x = (target_keypoints['left_hip'][1] +
                target_keypoints['right_hip'][1]) / 2;

    (max_torso_yrange, max_torso_xrange,
      max_body_yrange, max_body_xrange) = determine_torso_and_body_range(
          keypoints, target_keypoints, center_y, center_x)

    crop_length_half = np.amax(
        [max_torso_xrange * 1.9, max_torso_yrange * 1.9,
          max_body_yrange * 1.2, max_body_xrange * 1.2])

    tmp = np.array(
        [center_x, image_width - center_x, center_y, image_height - center_y])
    crop_length_half = np.amin(
        [crop_length_half, np.amax(tmp)]);

    crop_corner = [center_y - crop_length_half, center_x - crop_length_half];

    if crop_length_half > max(image_width, image_height) / 2:
      return init_crop_region(image_height, image_width)
    else:
      crop_length = crop_length_half * 2;
      return {
        'y_min': crop_corner[0] / image_height,
        'x_min': crop_corner[1] / image_width,
        'y_max': (crop_corner[0] + crop_length) / image_height,
        'x_max': (crop_corner[1] + crop_length) / image_width,
        'height': (crop_corner[0] + crop_length) / image_height -
            crop_corner[0] / image_height,
        'width': (crop_corner[1] + crop_length) / image_width -
            crop_corner[1] / image_width
      }
  else:
    return init_crop_region(image_height, image_width)

def crop_and_resize(image,crop_region,input_size):

  image_height, image_width, _ = image.shape

  ymin = int( crop_region['y_min'] * image_height)
  xmin = int( crop_region['x_min'] * image_width  )
  ymax = int( crop_region['y_max'] * image_height  )
  xmax = int( crop_region['x_max'] * image_width  )
  h = crop_region['height']
  w = crop_region['width']

  if ymin < 0 and xmin < 0:
    yp,xp = np.absolute(ymin),np.absolute(xmin)
    padding_image = np.zeros((2*yp+image_height,2*xp+image_width,3))
    padding_image[yp:-yp,xp:-xp,:] = image[:,:,:]
    input_image = padding_image[0:ymax+yp,0:xmax+xp]
  elif ymin < 0:
    yp = np.absolute(ymin)
    padding_image = np.zeros((2*yp+image_height,image_width,3))
    padding_image[yp:-yp,:,:] = image[:,:,:]
    input_image = padding_image[0:ymax+yp,xmin:xmax]
  elif xmin < 0:
    xp = np.absolute(xmin)
    padding_image = np.zeros((image_height,2*xp+image_width,3))
    padding_image[:,xp:-xp,:] = image[:,:,:]
    input_image = padding_image[ymin:ymax,0:xmax+xp]
  else:
    input_image = image[ymin:ymax,xmin:xmax]
  
  input_image = cv2.resize( input_image , (input_size,input_size))

  return input_image
