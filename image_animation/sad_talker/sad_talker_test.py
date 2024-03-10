from PIL import Image
from image_processor import FaceDetector, KeypointExtractor, ImageProcessor
from audio_processor import AudioProcessor
import cv2

IMAGE_PATH = "/home/t-ibayashi/Workspace/ax/repos/SadTalker/examples/source_image/art_5.png"
AUDIO_PATH = "/home/t-ibayashi/Workspace/ax/repos/SadTalker/examples/driven_audio/imagine.wav"
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

def test_main():
    # params
    input_image_path = IMAGE_PATH
    input_audio_path = AUDIO_PATH
    output_dir = "./results"
    ref_eyeblink = None
    ref_pose = None

    # 1. extract coefficients from image
    image_procesor = ImageProcessor()
    print('3DMM Extraction for source image')

    first_coeff_path, crop_pic_path, crop_info = image_procesor.generate(
        input_path=input_image_path,
        preprocess="resize",
        save_dir=output_dir,
        input_size=256,
    )
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return

    if ref_eyeblink is not None:
        raise Exception("eye blink alignment is not supported yet")
    else:
        ref_eyeblink_coeff_path=None

    if ref_pose is not None:
        raise Exception("pose alignment is not supported yet")

    # 2. extract coefficients from audio
    audio_processor = AudioProcessor()
    audio_processor.generate_batch(
        first_coeff_path,
        input_audio_path,
        device="cpu",
    )
    import pdb; pdb.set_trace()

import random
random.seed(314)

if __name__ == '__main__':
    # test_preprocess()
    test_main()
