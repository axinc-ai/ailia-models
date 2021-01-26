import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.special import expit
from scipy.ndimage.filters import gaussian_filter

EFFICIENT_POSE_KEYPOINT_HEAD_TOP   	       = (0)
EFFICIENT_POSE_KEYPOINT_UPPER_NECK	       = (1)
EFFICIENT_POSE_KEYPOINT_RIGHT_SHOULDER	   = (2)
EFFICIENT_POSE_KEYPOINT_RIGHT_ELBOW	       = (3)
EFFICIENT_POSE_KEYPOINT_RIGHT_WRIST	       = (4)
EFFICIENT_POSE_KEYPOINT_THORAX			   = (5)
EFFICIENT_POSE_KEYPOINT_LEFT_SHOULDER	   = (6)
EFFICIENT_POSE_KEYPOINT_LEFT_ELBOW		   = (7)
EFFICIENT_POSE_KEYPOINT_LEFT_RIGHT_WRIST   = (8)
EFFICIENT_POSE_KEYPOINT_PELVIS		       = (9)
EFFICIENT_POSE_KEYPOINT_RIGHT_HIP		   = (10)
EFFICIENT_POSE_KEYPOINT_RIGHT_KNEE		   = (11)
EFFICIENT_POSE_KEYPOINT_RIGHT_ANKLE	       = (12)
EFFICIENT_POSE_KEYPOINT_LEFT_HIP		   = (13)
EFFICIENT_POSE_KEYPOINT_LEFT_KNEE		   = (14)
EFFICIENT_POSE_KEYPOINT_LEFT_ANKLE		   = (15)

EFFICIENT_POSE_KEYPOINT_CNT = 16

def resize(source_array, target_height, target_width):
    """ 
    Resizes an image or image-like Numpy array to be no larger than (target_height, target_width) or (target_height, target_width, c).
    
    Args:
        source_array: ndarray
            Numpy array of shape (h, w) or (h, w, 3)
        target_height: int
            Desired maximum height
        target_width: int
            Desired maximum width
        
    Returns:
        Resized Numpy array.
    """
    
    # Get height and width of source array
    source_height, source_width = source_array.shape[:2]
     
    # Compute correct scale for resizing operation
    target_ratio = target_height / target_width
    source_ratio = source_height / source_width
    if target_ratio > source_ratio:
        scale = target_width / source_width
    else:
        scale = target_height / source_height
        
    # Perform rescaling
    width = int(source_width * scale)
    height = int(source_height * scale)
    dim = (width, height)
    # resize image
    resized_array = cv2.resize(source_array, dim, interpolation = cv2.INTER_AREA)/255.

    # resized_array = rescale(source_array, scale, multichannel=True)
    
    return resized_array

def pad(source_array, target_height, target_width):
    """ 
    Pads an image or image-like Numpy array with zeros to fit the target-size.
    
    Args:
        source_array: ndarray
            Numpy array of shape (h, w) or (h, w, 3)
        target_height: int
            Height of padded image
        target_width: int
            Width of padded image
    
    Returns:
        Zero-padded Numpy array of shape (target_height, target_width) or (target_height, target_width, c).
    """
    
    # Get height and width of source array
    source_height, source_width = source_array.shape[:2]
    
    # Ensure array is resized properly
    if (source_height > target_height) or (source_width > target_width):
        source_array = resize(source_array, target_height, target_width)
        source_height, source_width = source_array.shape[:2]
        
    # Compute padding variables
    pad_left = int((target_width - source_width) / 2)
    pad_top = int((target_height - source_height) / 2)
    pad_right = int(target_width - source_width - pad_left)
    pad_bottom = int(target_height - source_height - pad_top)
    paddings = [[pad_top, pad_bottom], [pad_left, pad_right]]
    has_channels_dim = len(source_array.shape) == 3
    if has_channels_dim:  
        paddings.append([0,0])
        
    # Perform padding
    target_array = np.pad(source_array, paddings, 'constant')
    
    return target_array

def normalize(x):
    """Preprocesses a Numpy array encoding a batch of images.
    Arguments:
        x: Input array, 3D or 4D.

    Returns:
        Preprocessed Numpy array.
    """

    x /= 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Zero-center by mean pixel
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]
    if std is not None:
        x[..., 0] /= std[0]
        x[..., 1] /= std[1]
        x[..., 2] /= std[2]

    return x

def preprocess(batch, resolution, lite=False):
    """
    Preprocess Numpy array according to model preferences.
    
    Args:
        batch: ndarray
            Numpy array of shape (n, h, w, 3)
        resolution: int
            Input height and width of model to utilize
        lite: boolean
            Defines if EfficientPose Lite model is used
    
    Returns:
        Preprocessed Numpy array of shape (n, resolution, resolution, 3).
    """
    
    batch = [resize(frame, resolution, resolution) for frame in batch]

    # Pad frames in batch to form quadratic input
    batch = [pad(frame, resolution, resolution) for frame in batch]

    # Convert from normalized pixels to RGB absolute values
    batch = [np.uint8(255 * frame) for frame in batch]

    # Construct Numpy array from batch
    batch = np.asarray(batch).astype('float')

    # Normalize images in batch
    batch = normalize(batch)


    return batch

def extract_coordinates(frame_output, frame_height, frame_width, real_time=False):
    """
    Extract coordinates from supplied confidence maps.
    
    Args:
        frame_output: ndarray
            Numpy array of shape (h, w, c)
        frame_height: int
            Height of relevant frame
        frame_width: int
            Width of relevant frame
        real-time: boolean
            Defines if processing is performed in real-time
           
    Returns:
        List of predicted coordinates for all c body parts in the frame the outputs are computed from.
    """
    
    # Define body parts
    body_parts = ['head_top', 'upper_neck', 'right_shoulder', 'right_elbow', 'right_wrist', 'thorax', 'left_shoulder', 'left_elbow', 'left_wrist', 'pelvis', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle']
    
    # Define confidence level
    confidence = 0.3
    
    # Fetch output resolution 
    output_height, output_width = frame_output.shape[0:2]
    
    # Initialize coordinates
    frame_coords = []
    
    # Iterate over body parts
    for i in range(frame_output.shape[-1]):

        # Find peak point
        conf = frame_output[...,i]
        if not real_time:
            conf = gaussian_filter(conf, sigma=1.) 
        max_index = np.argmax(conf)
        peak_y = float(math.floor(max_index / output_width))
        peak_x = max_index % output_width
        conf_xy = conf[int(peak_y),int(peak_x)]
        # Verify confidence
        if real_time and conf_xy < confidence:
            peak_x = -0.5
            peak_y = -0.5
        else:
            peak_x += 0.5
            peak_y += 0.5

        # Normalize coordinates
        peak_x /= output_width
        peak_y /= output_height

        # Convert to original aspect ratio 
        if frame_width > frame_height:
            norm_padding = (frame_width - frame_height) / (2 * frame_width)  
            peak_y = (peak_y - norm_padding) / (1.0 - (2 * norm_padding))
            peak_y = -0.5 / output_height if peak_y < 0.0 else peak_y
            peak_y = 1.0 if peak_y > 1.0 else peak_y
        elif frame_width < frame_height:
            norm_padding = (frame_height - frame_width) / (2 * frame_height)  
            peak_x = (peak_x - norm_padding) / (1.0 - (2 * norm_padding))
            peak_x = -0.5 / output_width if peak_x < 0.0 else peak_x
            peak_x = 1.0 if peak_x > 1.0 else peak_x

        frame_coords.append((body_parts[i], peak_x, peak_y, conf_xy))
        
    return frame_coords

def display_body_parts(image, image_draw, coordinates, image_height=1024, image_width=1024, marker_radius=5):   
    """
    Draw markers on predicted body part locations.
    
    Args:
        image: PIL Image
            The loaded image the coordinate predictions are inferred for
        image_draw: PIL ImageDraw module
            Module for performing drawing operations
        coordinates: List
            Predicted body part coordinates in image
        image_height: int
            Height of image
        image_width: int
            Width of image
        marker_radius: int
            Radius of marker
           
    Returns:
        Instance of PIL image with annotated body part predictions.
    """
    
    # Define body part colors
    body_part_colors = ['#fff142', '#fff142', '#576ab1', '#5883c4', '#56bdef', '#f19718', '#d33592', '#d962a6', '#e18abd', '#f19718', '#8ac691', '#a3d091', '#bedb8f', '#7b76b7', '#907ab8', '#a97fb9']
    
    # Draw markers
    for i, (body_part, body_part_x, body_part_y, conf) in enumerate(coordinates):
        body_part_x *= image_width
        body_part_y *= image_height
        image_draw.ellipse([(body_part_x - marker_radius, body_part_y - marker_radius), (body_part_x + marker_radius, body_part_y + marker_radius)], fill=body_part_colors[i])
        
    return image

def display_segments(image, image_draw, coordinates, image_height=1024, image_width=1024, segment_width=5):
    """
    Draw segments between body parts according to predicted body part locations.
    
    Args:
        image: PIL Image
            The loaded image the coordinate predictions are inferred for
        image_draw: PIL ImageDraw module
            Module for performing drawing operations
        coordinates: List
            Predicted body part coordinates in image
        image_height: int
            Height of image
        image_width: int
            Width of image
        segment_width: int
            Width of association line between markers
           
    Returns:
        Instance of PIL image with annotated body part segments.
    """
   
    # Define segments and colors
    segments = [(0, 1), (1, 5), (5, 2), (5, 6), (5, 9), (2, 3), (3, 4), (6, 7), (7, 8), (9, 10), (9, 13), (10, 11), (11, 12), (13, 14), (14, 15)]
    segment_colors = ['#fff142', '#fff142', '#576ab1', '#5883c4', '#56bdef', '#f19718', '#d33592', '#d962a6', '#e18abd', '#f19718', '#8ac691', '#a3d091', '#bedb8f', '#7b76b7', '#907ab8', '#a97fb9']
    
    # Draw segments
    for (body_part_a_index, body_part_b_index) in segments:
        _, body_part_a_x, body_part_a_y, conf_a = coordinates[body_part_a_index]
        body_part_a_x *= image_width
        body_part_a_y *= image_height
        _, body_part_b_x, body_part_b_y, conf_b = coordinates[body_part_b_index]
        body_part_b_x *= image_width
        body_part_b_y *= image_height
        image_draw.line([(body_part_a_x, body_part_a_y), (body_part_b_x, body_part_b_y)], fill=segment_colors[body_part_b_index], width=segment_width)
    
    return image

def display_camera(cv2, frame, coordinates, frame_height, frame_width):
    """
    Display camera frame with annotated body parts and segments according to predicted body part locations.
    
    Args:
        cv2: OpenCV
            Imported OpenCV instance
        frame: ndarray
            Numpy array of shape (h, w, 3)
        coordinates: List
            Predicted body part coordinates in frame
        frame_height: int
            Height of frame
        frame_width: int
            Width of frame
    """
    
    # Define body parts and segments
    segments = [(0, 1), (1, 5), (5, 2), (5, 6), (5, 9), (2, 3), (3, 4), (6, 7), (7, 8), (9, 10), (9, 13), (10, 11), (11, 12), (13, 14), (14, 15)]
    body_part_colors = [(66, 241, 255), (66, 241, 255), (177, 106, 87), (196, 131, 88), (239, 189, 86), (24, 151, 241), (146, 53, 211), (166, 98, 217), (189, 138, 225), (24, 151, 241), (145, 198, 138), (145, 208, 163), (143, 219, 190), (183, 118, 123), (184, 122, 144), (185, 127, 169)]
    
    # Draw lines and markers
    remaining = [i for i in range(len(body_part_colors))]
    for (a, b) in segments:
        a_coordinates = coordinates[a]
        a_coordinate_x = int(a_coordinates[1] * frame_width)
        a_coordinate_y = int(a_coordinates[2] * frame_height)
        b_coordinates = coordinates[b]
        b_coordinate_x = int(b_coordinates[1] * frame_width)
        b_coordinate_y = int(b_coordinates[2] * frame_height)
        if not (a_coordinate_x < 0 or a_coordinate_y < 0 or b_coordinate_x < 0 or b_coordinate_y < 0): 
            cv2.line(frame, (a_coordinate_x, a_coordinate_y), (b_coordinate_x, b_coordinate_y), color=body_part_colors[a], thickness=2)

            if a in remaining:
                cv2.circle(frame, (a_coordinate_x, a_coordinate_y), radius=3, color=body_part_colors[a], thickness=2)
                remaining.remove(a)
            if b in remaining:
                cv2.circle(frame, (b_coordinate_x, b_coordinate_y), radius=3, color=body_part_colors[b], thickness=2)
                remaining.remove(b)
                    
    # Display predictions
    frame = cv2.resize(cv2.flip(frame, 1), (1000, 1000))
    cv2.imshow('EfficientPose (Groos et al., 2020)', frame)