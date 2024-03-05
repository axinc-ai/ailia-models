import cv2
import sys
import numpy as np
from dataclasses import dataclass

sys.path.append('../../face_detection/retinaface')
from retinaface import postprocessing
import retinaface_utils as rut

sys.path.append('../../util')
from image_utils import normalize_image

def prep_input_numpy(img:np.ndarray):
    """Preparing a Numpy Array as input to L2CS-Net."""

    imgs = []
    for im in img:
        im = cv2.resize(im,(448,448))
        im = normalize_image(im,normalize_type="ImageNet")
        im = np.transpose(im,(2,0,1))
        imgs.append(im)
    img = np.stack(imgs)

    return img
  
@dataclass
class GazeResultContainer:
    pitch: np.ndarray
    yaw: np.ndarray
    bboxes: np.ndarray
    landmarks: np.ndarray
    scores: np.ndarray

class l2cs:

    def __init__(self, object_detection,l2cs, confidence_threshold:float = 0.5):
        # Save input parameters
        self.confidence_threshold = confidence_threshold
        self.object_detection_model = object_detection
        self.l2cs_model = l2cs

    def step(self, frame: np.ndarray) -> GazeResultContainer:

        # Creating containers
        face_imgs = []
        bboxes = []
        landmarks = []
        scores = []

        IMAGE_WIDTH = int(frame.shape[1])
        IMAGE_HEIGHT = int(frame.shape[0])
        dim = (IMAGE_WIDTH, IMAGE_HEIGHT)
        org_img = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

        img = org_img - (104, 117, 123)
        input_data = img.transpose(2, 0, 1)
        input_data.shape = (1,) + input_data.shape

        cfg = rut.cfg_re50
        faces = self.object_detection_model.predict([input_data])
        faces = postprocessing(faces,input_data,cfg,dim)

        if faces is not None: 
            #for box, landmark, score in faces:
            for face in faces:
                box = face[:5]
                landmark = face[6:]
                score = face[5]

                # Apply threshold
                if score < self.confidence_threshold:
                    continue

                # Extract safe min and max of x,y
                x_min=int(box[0])
                if x_min < 0:
                    x_min = 0
                y_min=int(box[1])
                if y_min < 0:
                    y_min = 0
                x_max=int(box[2])
                y_max=int(box[3])
                
                # Crop image
                img = frame[y_min:y_max, x_min:x_max]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                face_imgs.append(img)

                # Save data
                bboxes.append(box)
                landmarks.append(landmark)
                scores.append(score)

            # Predict gaze
            pitch, yaw = self.predict_gaze(np.stack(face_imgs))

        else:
            pitch = np.empty((0,1))
            yaw = np.empty((0,1))

        # Save data
        results = GazeResultContainer(
            pitch=pitch,
            yaw=yaw,
            bboxes=np.stack(bboxes),
            landmarks=np.stack(landmarks),
            scores=np.stack(scores)
        )

        return results

    def predict_gaze(self, frame):
        
        # Prepare input
        img = prep_input_numpy(frame)
    
        # Predict 
        pitch_predicted ,yaw_predicted = self.l2cs_model.run(img)

        pitch_predicted= pitch_predicted* np.pi/180.0
        yaw_predicted= yaw_predicted * np.pi/180.0
        return pitch_predicted, yaw_predicted


def draw_gaze(a,b,c,d,image_in, pitchyaw, thickness=2, color=(255, 255, 0),sclae=2.0):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    (h, w) = image_in.shape[:2]
    length = c
    pos = (int(a+c / 2.0), int(b+d / 2.0))
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[0]) * np.cos(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[1])
    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                   tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.18)
    return image_out

def draw_bbox(frame: np.ndarray, bbox: np.ndarray):
    
    x_min=int(bbox[0])
    if x_min < 0:
        x_min = 0
    y_min=int(bbox[1])
    if y_min < 0:
        y_min = 0
    x_max=int(bbox[2])
    y_max=int(bbox[3])

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)

    return frame

def render(frame: np.ndarray, results: GazeResultContainer):

    # Draw bounding boxes
    for bbox in results.bboxes:
        frame = draw_bbox(frame, bbox)

    # Draw Gaze
    for i in range(results.pitch.shape[0]):

        bbox = results.bboxes[i]
        pitch = results.pitch[i]
        yaw = results.yaw[i]
        
        # Extract safe min and max of x,y
        x_min=int(bbox[0])
        if x_min < 0:
            x_min = 0
        y_min=int(bbox[1])
        if y_min < 0:
            y_min = 0
        x_max=int(bbox[2])
        y_max=int(bbox[3])

        # Compute sizes
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        draw_gaze(x_min,y_min,bbox_width, bbox_height,frame,(pitch,yaw),color=(0,0,255))

    return frame
