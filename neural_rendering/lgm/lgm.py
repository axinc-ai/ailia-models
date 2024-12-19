import sys
import numpy as np
import imageio.v2 as imageio
import kiui
from kiui.cam import orbit_camera
import tqdm
import transformers

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models, check_and_download_file  # noqa

# logger
from logging import getLogger  # noqa
logger = getLogger(__name__)

from df.pipelines.pipeline_mvdream import MVDreamPipeline
from df.schedulers.scheduling_ddim import DDIMScheduler
from utils import preprocess_mvdream_pipeline, preprocess_lgm_model
from gs import GaussianRenderer

# ======================
# Parameters
# ======================
WEIGHT_UNET_PATH = "unet.onnx"
# MODEL_UNET_PATH = "unet.onnx.prototxt"
WEIGHT_TEXT_ENCODER_PATH = "text_encoder.onnx"
# MODEL_TEXT_ENCODER_PATH = "text_encoder.onnx.prototxt"
WEIGHT_VAE_DECODER_PATH = "vae_decoder.onnx"
# MODEL_VAE_DECODER_PATH = "vae_decoder.onnx.prototxt"
WEIGHT_VAE_ENCODER_PATH = "vae_encoder.onnx"
# MODEL_VAE_ENCODER_PATH = "vae_encoder.onnx.prototxt"
WEIGHT_IMAGE_ENCODER_PATH = "image_encoder.onnx"
# MODEL_IMAGE_ENCODER_PATH = "image_encoder.onnx.prototxt"
WEIGHT_LGM_PATH = "lgm.onnx"
# MODEL_LGM_PATH = "lgm.onnx.prototxt"

MODEL_UNET_PATH = None
MODEL_TEXT_ENCODER_PATH = None
MODEL_VAE_DECODER_PATH = None
MODEL_VAE_ENCODER_PATH = None
MODEL_IMAGE_ENCODER_PATH = None
MODEL_LGM_PATH = None

REMOTE_PATH = "https://storage.googleapis.com/ailia-models/lgm/"

INPUT_SIZE = 256
OUTPUT_SIZE = 256
FOVY = 49.1
CAM_RADIUS = 1.5
ZFAR = 2.5
ZNEAR = 0.5
IMAGE_PATH = 'catstatue_rgba.png'
SAVE_VIDEO_PATH = 'output.gif'

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'lgm', IMAGE_PATH, SAVE_VIDEO_PATH
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="random seed",
)
parser.add_argument(
    "--output_size",
    type=int,
    default=None,
    help="output size",
)
parser.add_argument(
    '--disable_ailia_tokenizer',
    action='store_true',
    help='disable ailia tokenizer.'
)
parser.add_argument(
    '-o', '--onnx', action='store_true',
    help="Option to use onnxrutime to run or not."
)
args = update_parser(parser)

# ======================
# Utils
# ======================
class Options:
    input_size: int = INPUT_SIZE
    output_size: int = args.output_size if args.output_size is not None else OUTPUT_SIZE
    fovy: float = FOVY
    cam_radius: float = CAM_RADIUS
    zfar: float = ZFAR
    znear: float = ZNEAR

opt = Options()

def load_model(weight_path, model_path, env_id=None, memory_mode=None, use_onnx=False):
    if use_onnx:
        import onnxruntime
        cuda = 0 < ailia.get_gpu_environment_id()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if cuda
            else ["CPUExecutionProvider"]
        )
        return onnxruntime.InferenceSession(weight_path, providers=providers)
    else:
        return ailia.Net(model_path, weight_path, env_id=env_id, memory_mode=memory_mode)

def generate_multi_view_images(image):
    env_id = args.env_id
    user_onnx = args.onnx

    model_paths = [
        (WEIGHT_UNET_PATH, MODEL_UNET_PATH),
        (WEIGHT_TEXT_ENCODER_PATH, MODEL_TEXT_ENCODER_PATH),
        (WEIGHT_VAE_ENCODER_PATH, MODEL_VAE_ENCODER_PATH),
        (WEIGHT_VAE_DECODER_PATH, MODEL_VAE_DECODER_PATH),
        (WEIGHT_IMAGE_ENCODER_PATH, MODEL_IMAGE_ENCODER_PATH),
    ]
    for weight, model in model_paths:
        check_and_download_models(weight, model, REMOTE_PATH)

    memory_mode = None
    if not user_onnx:
        memory_mode = ailia.get_memory_mode(
            reduce_constant=True,
            ignore_input_with_initializer=True,
            reduce_interstage=False,
            reuse_interstage=True,
        )
    unet = load_model(WEIGHT_UNET_PATH, MODEL_UNET_PATH, env_id, memory_mode, user_onnx)
    text_encoder = load_model(
        WEIGHT_TEXT_ENCODER_PATH, MODEL_TEXT_ENCODER_PATH, env_id, memory_mode, user_onnx
    )
    vae_encoder = load_model(
        WEIGHT_VAE_ENCODER_PATH, MODEL_VAE_ENCODER_PATH, env_id, memory_mode, user_onnx
    )
    vae_decoder = load_model(
        WEIGHT_VAE_DECODER_PATH, MODEL_VAE_DECODER_PATH, env_id, memory_mode, user_onnx
    )
    image_encoder = load_model(
        WEIGHT_IMAGE_ENCODER_PATH, MODEL_IMAGE_ENCODER_PATH, env_id, memory_mode, user_onnx
    )

    if args.disable_ailia_tokenizer:
        tokenizer = transformers.CLIPTokenizer.from_pretrained("./tokenizer")
    else:
        from ailia_tokenizer import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained()
        tokenizer.model_max_length = 77
        tokenizer.add_special_tokens({'pad_token': '!'})

    feature_extractor = transformers.CLIPImageProcessor.from_pretrained("./feature_extractor")

    scheduler = DDIMScheduler.from_config({
        "prediction_type": "epsilon",
        "clip_sample": False,
        "beta_schedule": "scaled_linear",
        "beta_start": 0.00085,
        "set_alpha_to_one": False,
        "trained_betas": None,
        "steps_offset": 1,
        "beta_end": 0.012,
        "num_train_timesteps": 1000,
        "clip_sample_range": 1.0,
        "dynamic_thresholding_ratio": 0.995,
        "thresholding": False,
        "timestep_spacing": "leading",
    })

    mvdream_pipe = MVDreamPipeline(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        vae_encoder=vae_encoder,
        vae_decoder=vae_decoder,
        unet=unet,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        scheduler=scheduler,
        use_onnx=args.onnx,
    )

    logger.info("Generating multi-view images...")
    mv_image = mvdream_pipe('', image, guidance_scale=5.0, num_inference_steps=30, elevation=0)
    mv_image = np.stack([mv_image[1], mv_image[2], mv_image[3], mv_image[0]], axis=0) # [4, 256, 256, 3]
    return mv_image

def generate_gaussians(mv_image):
    check_and_download_models(WEIGHT_LGM_PATH, MODEL_LGM_PATH, REMOTE_PATH)
    lgm = load_model(WEIGHT_LGM_PATH, MODEL_LGM_PATH, args.env_id, use_onnx=args.onnx)

    if args.onnx:
        gaussians = lgm.run(None, {lgm.get_inputs()[0].name: mv_image})[0]
    else:
        gaussians = lgm.predict(mv_image)

    return gaussians

def render_video(gaussians, opt):
    gs = GaussianRenderer(opt)

    tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
    proj_matrix = np.zeros((4, 4), dtype=np.float32)
    proj_matrix[0, 0] = 1 / tan_half_fov
    proj_matrix[1, 1] = 1 / tan_half_fov
    proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[2, 3] = 1

    images = []
    elevation = 0
    
    azimuth = np.arange(0, 360, 6, dtype=np.int32)

    logger.info("Rendering video...")
    for azi in tqdm.tqdm(azimuth):
        cam_poses = np.expand_dims(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True), axis=0)
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction

        cam_view = np.linalg.inv(cam_poses).transpose(0, 2, 1) # [V, 4, 4]
        cam_view_proj = np.matmul(cam_view, proj_matrix) # [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]

        image = gs.render(
            gaussians, 
            np.expand_dims(cam_view, axis=0), 
            np.expand_dims(cam_view_proj, axis=0), 
            np.expand_dims(cam_pos, axis=0), 
            scale_modifier=1
        )
        images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().numpy() * 255).astype(np.uint8))

    images = np.concatenate(images, axis=0)
    return images

# ======================
# Main functions
# ======================
def main():
    seed = args.seed
    if seed is not None:
        np.random.seed(seed)

    # load image
    input_image = kiui.read_image(args.input[0], mode='uint8')

    # generate multi-view images
    processed_image = preprocess_mvdream_pipeline(input_image)
    multi_view_image = generate_multi_view_images(processed_image)

    # generate gaussians
    processed_mv_image = preprocess_lgm_model(multi_view_image, opt)
    gaussians = generate_gaussians(processed_mv_image)

    # render video
    images = render_video(gaussians, opt)

    # save result
    save_path = get_savepath(args.savepath, "", ext=".gif")
    imageio.mimsave(save_path, images, fps=30, loop=0)

    logger.info("Script finished successfully.")

if __name__ == '__main__':
    main()
