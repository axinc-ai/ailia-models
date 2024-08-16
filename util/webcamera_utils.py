import os
import sys
import re

import numpy as np
import cv2

from arg_utils import check_file_existance
from image_utils import normalize_image

from logging import getLogger
logger = getLogger(__name__)


def calc_adjust_fsize(f_height, f_width, height, width):
    # calculate the image size of the output('img') of adjust_frame_size
    # This function is supposed to be used to declare 'cv2.writer'
    scale = np.max((f_height / height, f_width / width))
    return int(scale * height), int(scale * width)


def adjust_frame_size(frame, height, width):
    """
    Adjust the size of the frame from the webcam to the ailia input shape.

    Parameters
    ----------
    frame: numpy array
    height: int
        ailia model input height
    width: int
        ailia model input width

    Returns
    -------
    img: numpy array
        Image with the propotions of height and width
        adjusted by padding for ailia model input.
    resized_img: numpy array
        Resized `img` as well as adapt the scale
    """
    f_height, f_width = frame.shape[0], frame.shape[1]
    scale = np.max((f_height / height, f_width / width))

    # padding base
    img = np.zeros(
        (int(round(scale * height)), int(round(scale * width)), 3),
        np.uint8
    )
    start = (np.array(img.shape) - np.array(frame.shape)) // 2
    img[
        start[0]: start[0] + f_height,
        start[1]: start[1] + f_width
    ] = frame
    resized_img = cv2.resize(img, (width, height))
    return img, resized_img


def cut_max_square(frame: np.array) -> np.array:
    """
    Cut out a maximum square area from the center of given frame (np.array).
    Parameters
    ----------
    frame: numpy array

    Returns
    -------
    frame_square: numpy array
        Maximum square area of the frame at its center
    """
    frame_height, frame_width, _ = frame.shape
    frame_size_min = min(frame_width, frame_height)
    if frame_width >= frame_height:
        x, y = frame_width // 2 - frame_height // 2, 0
    else:
        x, y = 0, frame_height // 2 - frame_width // 2

    frame_square = frame[y: (y + frame_size_min), x: (x + frame_size_min)]
    return frame_square


def preprocess_frame(
        frame, input_height, input_width, data_rgb=True, normalize_type='255'
):
    """
    Pre-process the frames taken from the webcam to input to ailia.

    Parameters
    ----------
    frame: numpy array
    input_height: int
        ailia model input height
    input_width: int
        ailia model input width
    data_rgb: bool (default: True)
        Convert as rgb image when True, as gray scale image when False.
        Only `data` will be influenced by this configuration.
    normalize_type: string (default: 255)
        Normalize type should be chosen from the type below.
        - '255': simply dividing by 255.0
        - '127.5': output range : -1 and 1
        - 'ImageNet': normalize by mean and std of ImageNet
        - 'None': no normalization

    Returns
    -------
    img: numpy array
        Image with the propotions of height and width
        adjusted by padding for ailia model input.
    data: numpy array
        Input data for ailia
    """
    img, resized_img = adjust_frame_size(frame, input_height, input_width)

    if data_rgb:
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    data = normalize_image(resized_img, normalize_type)

    if data_rgb:
        data = np.rollaxis(data, 2, 0)
        data = np.expand_dims(data, axis=0).astype(np.float32)
    else:
        data = cv2.cvtColor(data.astype(np.float32), cv2.COLOR_BGR2GRAY)
        data = data[np.newaxis, np.newaxis, :, :]
    return img, data


def get_writer(savepath, height, width, fps=20, rgb=True):
    """get cv2.VideoWriter

    Parameters
    ----------
    save_path : str
    height : int
    width : int
    fps : int
    rgb : bool, default is True

    Returns
    -------
    writer : cv2.VideoWriter()
    """
    # stream output
    if re.match(r'localhost\:',savepath) or re.match(r'[0-9]+(?:\.[0-9]+){3}\:',savepath):
        # Usage : 
        #   server : python3 xxx.py -v 0 -s localhost:5000
        #   client : gst-launch-1.0 -v tcpclientsrc host=localhost port=5000 ! gdpdepay ! videoconvert ! autovideosink sync=false

        bitrate = 10000000
        tcp = True
        ip = savepath.split(":")[0]
        port = savepath.split(":")[1]
        logger.info("gstreamer open with ip "+str(ip)+" port "+str(port))
        encoder = 'nvvidconv ! nvv4l2h264enc bitrate=' + str(bitrate) + ' insert-sps-pps=true maxperf-enable=1'
        sink =  'appsrc ! video/x-raw,format=BGR ! queue ! videoconvert ! video/x-raw,format=BGRx ! ' + encoder + ' ! rtph264pay config-interval=1 ! ' + ('gdppay ! tcpserversink' if tcp else 'udpsink') + ' host=' + str(ip) + ' port=' + str(port)
        writer = cv2.VideoWriter(sink, 0, int(fps), (width, height))
        if not writer.isOpened():
            logger.error(f"gstreamer could not open")
            sys.exit(0)
        return writer

    # file output
    if os.path.isdir(savepath):
        savepath = savepath + "/out.mp4"

    writer = cv2.VideoWriter(
        savepath,
        # cv2.VideoWriter_fourcc(*'MJPG'),  # avi mode
        cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),  # mp4 mode
        fps,
        (width, height),
        isColor=rgb
    )
    return writer


class BaslerCameraCapture:
    def __init__(self):
        self.camera = None
        self.converter = None

    def start_capture(self):
        from pypylon import pylon
        # Pylonカメラの初期化
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.Open()
        
        # キャプチャの設定
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.converter = pylon.ImageFormatConverter()

        # converting to opencv bgr format
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    def read(self):
        from pypylon import pylon
        if self.camera is None:
            raise Exception("Capture not started")

        grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grab_result.GrabSucceeded():
            converted_frame = self.converter.Convert(grab_result)
            rgb_frame = converted_frame.GetArray()
            grab_result.Release()
            return True, rgb_frame
        else:
            return False, None

    def stop_capture(self):
        if self.camera is not None:
            self.camera.Close()
            self.camera = None
            

def get_capture(video):
    """
    Get cv2.VideoCapture

    * TODO: maybe get capture & writer at the same time?
    *       then, you can use capture frame size directory

    Parameters
    ----------
    video : str
        webcamera-id or video path

    Returns
    -------
    capture : cv2.VideoCapture
    """
    try:
        video_id = int(video)

        # webcamera-mode
        capture = cv2.VideoCapture(video_id)
        if not capture.isOpened():
            logger.error(f"webcamera (ID - {video_id}) not found")
            sys.exit(0)

    except ValueError:
        # if file path is given, open video file
        if "rtsp://" in video:
            capture = cv2.VideoCapture(video)
        elif "basler" in video:
            capture = BaslerCameraCapture()
            capture.start_capture()
        elif check_file_existance(video):
            capture = cv2.VideoCapture(video)

    return capture
