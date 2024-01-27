from PIL import Image
from image_processor import FaceDetector, KeypointExtractor, ImageProcessor
import cv2

IMAGE_PATH = "/home/t-ibayashi/Workspace/ax/repos/SadTalker/examples/source_image/art_5.png"
PATH_TO_FACE_DETECTOR_ONNX = '/home/t-ibayashi/Workspace/ax/ailia-models/image_animation/sad_talker/resources/faceDetector.onnx'

def test_face_detector():
    pic_size=256
    full_frames = [cv2.imread(IMAGE_PATH)]
    x_full_frames= [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  for frame in full_frames]
    frames_pil = [Image.fromarray(cv2.resize(frame,(pic_size, pic_size))) for frame in x_full_frames]
    face_detector = FaceDetector(PATH_TO_FACE_DETECTOR_ONNX)
    face_detector.detect_face(frames_pil[0])

def test_keypoint_extractor():
    pic_size=256
    full_frames = [cv2.imread(IMAGE_PATH)]
    x_full_frames= [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  for frame in full_frames]
    frames_pil = [Image.fromarray(cv2.resize(frame,(pic_size, pic_size))) for frame in x_full_frames]
    keypoint_extractor = KeypointExtractor()
    keypoint_extractor.extract_keypoint(frames_pil[0], landmarks_path="./results/landmark")

def test_preprocess():
    model = ImageProcessor()
    result = model.generate(
        input_path=IMAGE_PATH,
        preprocess="resize",
        save_dir="results",
    )

if __name__ == '__main__':
    test_preprocess()
