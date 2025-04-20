from collections import OrderedDict
import os
import sys
from logging import getLogger

import numpy as np
from tqdm import tqdm

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from math_utils import softmax


logger = getLogger(__name__)


# ======================
# Parameters
# ======================

WEIGHT_BACKBONE_PATH = "backbone.onnx"
WEIGHT_SAM_PATH = "sam_heads.onnx"
WEIGHT_ENC_PATH = "memory_encoder.onnx"
MODEL_BACKBONE_PATH = "backbone.onnx.prototxt"
MODEL_SAM_PATH = "sam_heads.onnx.prototxt"
MODEL_ENC_PATH = "memory_encoder.onnx.prototxt"
REMOTE_PATH = "https://storage.googleapis.com/ailia-models/samurai/"

REF_WAV_PATH = "demo.mp4"
SAVE_WAV_PATH = "output.mp4"

IMG_SIZE = 1024

this_dir = os.path.dirname(os.path.abspath(__file__))


# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser("SAMURAI", None, SAVE_WAV_PATH)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser, check_input_type=False)


# ======================
# Secondary Functions
# ======================


def load_video_frames(video_path, image_size=None):
    return None, 1080, 1920


# ======================
# Main functions
# ======================


class SAM2VideoPredictor:
    """The predictor class to handle user interactions and manage inference states."""

    def __init__(self, backbone, sam_heads, memory_encoder, flg_onnx=False):
        self.backbone = backbone
        self.sam_heads = sam_heads
        self.memory_encoder = memory_encoder
        self.flg_onnx = flg_onnx

        self.num_feature_levels = 3
        self.hidden_dim = 256
        self.no_mem_embed = np.load(os.path.join(this_dir, "no_mem_embed.npy"))

    ### SAM2Base

    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        backbone_out = backbone_out.copy()

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # flatten NxCxHxW to HWxNxC
        vision_feats = [
            x.reshape(x.shape[0], x.shape[1], -1).transpose(2, 0, 1)
            for x in feature_maps
        ]
        vision_pos_embeds = [
            x.reshape(x.shape[0], x.shape[1], -1).transpose(2, 0, 1)
            for x in vision_pos_embeds
        ]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def _track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse,
        prev_sam_mask_logits,
    ):
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}
        # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
        high_res_features = [
            x.transpose(1, 2, 0).reshape(x.shape[1], x.shape[2], *s)
            for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
        ]

        if mask_inputs is not None:
            # When use_mask_input_as_output_without_sam=True, we directly output the mask input
            # (see it as a GT mask) without using a SAM prompt encoder + mask decoder.
            pix_feat = current_vision_feats[-1].transpose(1, 2, 0)
            pix_feat = pix_feat.reshape(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(
                pix_feat, high_res_features, mask_inputs
            )
        else:
            # fused the visual feature with previous memory features in the memory bank
            pix_feat = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats[-1:],
                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:],
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
            )
            # apply SAM-style segmentation head
            # here we might feed previously predicted low-res SAM mask logits into the SAM mask decoder,
            # e.g. in demo where such logits come from earlier interaction instead of correction sampling
            # (in this case, any `mask_inputs` shouldn't reach here as they are sent to _use_mask_as_output instead)
            if prev_sam_mask_logits is not None:
                mask_inputs = prev_sam_mask_logits
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)

            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
            )

        return current_out, sam_outputs, high_res_features, pix_feat

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        run_mem_encoder=True,
        prev_sam_mask_logits=None,
    ):
        current_out, sam_outputs, _, _ = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
        )

        (
            _,
            _,
            _,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
            best_iou_score,
            kf_ious,
        ) = sam_outputs

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        current_out["best_iou_score"] = best_iou_score
        current_out["kf_ious"] = kf_ious
        if not self.training:
            # Only add this in inference (to avoid unused param in activation checkpointing;
            # it's mainly used in the demo to encode spatial memories w/ consolidated masks)
            current_out["object_score_logits"] = object_score_logits

        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future frames)
        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
        )

        return current_out

    def _use_multimask(self, is_init_cond_frame, point_inputs):
        """Whether to use multimask output in the SAM head."""
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
        multimask_output = 0 <= num_pts <= 1

        return multimask_output

    ### VideoPredictor

    def init_state(self, images, video_height, video_width):
        self.images = images
        self.num_frames = len(images)
        self.video_height = video_height
        self.video_width = video_width

        # inputs on each frame
        self.point_inputs_per_obj = {}
        self.mask_inputs_per_obj = {}
        # visual features on a small number of recently visited frames for quick interactions
        self.cached_features = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        self.constants = {}
        # mapping between client-side object id and model-side object index
        self.obj_id_to_idx = OrderedDict()
        self.obj_idx_to_id = OrderedDict()
        self.obj_ids = []
        # A storage to hold the model's tracking results and states on each frame
        self.output_dict = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        self.output_dict_per_obj = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        self.temp_output_dict_per_obj = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        self.consolidated_frame_inds = {
            "cond_frame_outputs": set(),  # set containing frame indices
            "non_cond_frame_outputs": set(),  # set containing frame indices
        }
        # metadata for each tracking frame (e.g. which direction it's tracked)
        self.tracking_has_started = False
        self.frames_already_tracked = {}

    def _obj_id_to_idx(self, obj_id):
        """Map client-side object id to model-side object index."""
        obj_idx = self.obj_id_to_idx.get(obj_id, None)
        if obj_idx is not None:
            return obj_idx

        # This is a new object id not sent to the server before. We only allow adding
        # new objects *before* the tracking starts.
        if not self.tracking_has_started:
            # get the next object slot
            obj_idx = len(self.obj_id_to_idx)
            self.obj_id_to_idx[obj_id] = obj_idx
            self.obj_idx_to_id[obj_idx] = obj_id
            self.obj_ids = list(self.obj_id_to_idx)
            # set up input and output structures for this object
            self.point_inputs_per_obj[obj_idx] = {}
            self.mask_inputs_per_obj[obj_idx] = {}
            self.output_dict_per_obj[obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            self.temp_output_dict_per_obj[obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            return obj_idx
        else:
            raise RuntimeError(
                f"Cannot add new object id {obj_id} after tracking starts. "
                f"All existing object ids: {inference_state['obj_ids']}. "
                f"Please call 'reset_state' to restart from scratch."
            )

    def get_obj_num(self):
        return len(self.obj_idx_to_id)

    def add_new_points_or_box(
        self,
        frame_idx,
        obj_id,
        points=None,
        labels=None,
        clear_old_points=True,
        box=None,
    ):
        """Add new points to a frame."""
        obj_idx = self._obj_id_to_idx(obj_id)
        point_inputs_per_frame = self.point_inputs_per_obj[obj_idx]
        mask_inputs_per_frame = self.mask_inputs_per_obj[obj_idx]

        if points is None:
            points = np.zeros((0, 2), dtype=np.float32)
        else:
            points = np.array(points, dtype=np.float32)
        if labels is None:
            labels = np.zeros(0, dtype=np.int32)
        else:
            labels = np.array(labels, dtype=np.int32)
        if points.ndim == 2:
            points = np.expand_dims(points, axis=0)  # add batch dimension
        if labels.ndim == 1:
            labels = np.expand_dims(labels, axis=0)  # add batch dimension

        # If `box` is provided, we add it as the first two points with labels 2 and 3
        # along with the user-provided points (consistent with how SAM 2 is trained).
        box = np.array(box, dtype=np.float32)
        box_coords = box.reshape(1, 2, 2)
        box_labels = np.array([2, 3], dtype=np.int32)
        box_labels = box_labels.reshape(1, 2)
        points = box_coords
        labels = box_labels

        points = points / np.array([self.video_width, self.video_height])
        # scale the (normalized) coordinates by the model's internal image size
        points = points * IMG_SIZE

        if not clear_old_points:
            point_inputs = point_inputs_per_frame.get(frame_idx, None)
        else:
            point_inputs = None
        if point_inputs is not None:
            points = np.concatenate([point_inputs["point_coords"], points], axis=1)
            labels = np.concatenate([point_inputs["point_labels"], labels], axis=1)
        point_inputs = {"point_coords": points, "point_labels": labels}
        point_inputs_per_frame[frame_idx] = point_inputs
        mask_inputs_per_frame.pop(frame_idx, None)

        # If this frame hasn't been tracked before, we treat it as an initial conditioning
        # frame, meaning that the inputs points are to generate segments on this frame without
        # using any memory from other frames, like in SAM. Otherwise (if it has been tracked),
        # the input points will be used to correct the already tracked masks.
        is_init_cond_frame = frame_idx not in self.frames_already_tracked
        # whether to track in reverse time order
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = self.frames_already_tracked[frame_idx]["reverse"]

        obj_output_dict = self.output_dict_per_obj[obj_idx]
        obj_temp_output_dict = self.temp_output_dict_per_obj[obj_idx]
        # Add a frame to conditioning output if it's an initial conditioning frame or
        # if the model sees all frames receiving clicks/mask as conditioning frames.
        is_cond = is_init_cond_frame
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        # Get any previously predicted mask logits on this object and feed it along with
        # the new clicks into the SAM mask decoder.
        prev_sam_mask_logits = None
        # lookup temporary output dict first, which contains the most recent output
        # (if not found, then lookup conditioning and non-conditioning frame output)
        prev_out = obj_temp_output_dict[storage_key].get(frame_idx)
        if prev_out is None:
            prev_out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
            if prev_out is None:
                prev_out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)

        if prev_out is not None and prev_out["pred_masks"] is not None:
            prev_sam_mask_logits = prev_out["pred_masks"]
            # Clamp the scale of prev_sam_mask_logits to avoid rare numerical issues.
            prev_sam_mask_logits = np.clip(prev_sam_mask_logits, -32.0, 32.0)

        current_out, _ = self.single_frame_inference(
            output_dict=obj_output_dict,  # run on the slice of a single object
            frame_idx=frame_idx,
            batch_size=1,  # run on the slice of a single object
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=None,
            reverse=reverse,
            # Skip the memory encoder when adding clicks or mask. We execute the memory encoder
            # at the beginning of `propagate_in_video` (after user finalize their clicks). This
            # allows us to enforce non-overlapping constraints on all objects before encoding
            # them into memory.
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        # Add the output to the output dict (to be used as future memory)
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # Resize the output mask to the original video resolution
        obj_ids = self.obj_ids
        consolidated_out = self._consolidate_temp_output_across_obj(
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks

    def add_new_points(self, *args, **kwargs):
        """Deprecated method. Please use `add_new_points_or_box` instead."""
        return self.add_new_points_or_box(*args, **kwargs)

    def preflight(self):
        """Prepare inference_state and consolidate temporary outputs before tracking."""
        # Tracking has started and we don't allow adding new objects until session is reset.
        self.tracking_has_started = True

        batch_size = self.get_obj_num()

        for is_cond in [False, True]:
            # Separately consolidate conditioning and non-conditioning temp outputs
            storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

            temp_frame_inds = set()
            for obj_temp_output_dict in self.temp_output_dict_per_obj.values():
                temp_frame_inds.update(obj_temp_output_dict[storage_key].keys())

    def propagate_in_video(self):
        self.preflight()

        batch_size = self.get_obj_num()
        clear_non_cond_mem = False

        consolidated_frame_inds = {
            "cond_frame_outputs": {0},
            "non_cond_frame_outputs": {},
        }
        obj_ids = [0]

        start_frame_idx = 0
        end_frame_idx = 23
        processing_order = range(start_frame_idx, end_frame_idx + 1)

        for frame_idx in tqdm(processing_order, desc="propagate in video"):
            _, video_res_masks = self._get_orig_video_res_output(pred_masks)

            yield frame_idx, obj_ids, video_res_masks

    def add_output_per_object(self, frame_idx, current_out, storage_key):
        """
        Split a multi-object output into per-object output slices and add them into
        `output_dict_per_obj`. The resulting slices share the same tensor storage.
        """
        maskmem_features = current_out["maskmem_features"]
        maskmem_pos_enc = current_out["maskmem_pos_enc"]

        for obj_idx, obj_output_dict in self.output_dict_per_obj.items():
            obj_slice = slice(obj_idx, obj_idx + 1)
            obj_out = {
                "maskmem_features": None,
                "maskmem_pos_enc": None,
                "pred_masks": current_out["pred_masks"][obj_slice],
                "obj_ptr": current_out["obj_ptr"][obj_slice],
                "object_score_logits": current_out["object_score_logits"][obj_slice],
            }
            if maskmem_features is not None:
                obj_out["maskmem_features"] = maskmem_features[obj_slice]
            if maskmem_pos_enc is not None:
                obj_out["maskmem_pos_enc"] = [x[obj_slice] for x in maskmem_pos_enc]
            obj_output_dict[storage_key][frame_idx] = obj_out

    def get_image_feature(self, frame_idx, batch_size):
        """Compute the image features on a given frame."""
        # Look up in the cache first
        image, backbone_out = self.cached_features.get(frame_idx, (None, None))
        if backbone_out is None:
            image = np.expand_dims(self.images[frame_idx].astype(np.float32), axis=0)

            # feedforward
            if not self.flg_onnx:
                output = self.backbone.predict([image])
            else:
                output = self.backbone.run(None, {"image": image})
            (
                vision_features,
                vision_pos_enc_0,
                vision_pos_enc_1,
                vision_pos_enc_2,
                backbone_fpn_0,
                backbone_fpn_1,
                backbone_fpn_2,
            ) = output
            backbone_out = dict(
                vision_features=vision_features,
                vision_pos_enc=[vision_pos_enc_0, vision_pos_enc_1, vision_pos_enc_2],
                backbone_fpn=[backbone_fpn_0, backbone_fpn_1, backbone_fpn_2],
            )

            # Cache the most recent frame's feature (for repeated interactions with
            # a frame; we can use an LRU cache for more frames in the future).
            self.cached_features = {frame_idx: (image, backbone_out)}

        # expand the features to have the same dimension as the number of objects
        expanded_image = np.broadcast_to(image, (batch_size, *image.shape[1:]))
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = np.broadcast_to(
                feat, (batch_size, *feat.shape[1:])
            )
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            pos = np.broadcast_to(pos, (batch_size, *pos.shape[1:]))
            expanded_backbone_out["vision_pos_enc"][i] = pos

        features = self._prepare_backbone_features(expanded_backbone_out)
        features = (expanded_image,) + features
        return features

    def single_frame_inference(
        self,
        output_dict,
        frame_idx,
        batch_size,
        is_init_cond_frame,
        point_inputs,
        mask_inputs,
        reverse,
        run_mem_encoder,
        prev_sam_mask_logits=None,
    ):
        """Run tracking on a single frame based on current inputs and previous memory."""
        # Retrieve correct image features
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self.get_image_feature(frame_idx, batch_size)

        # point and mask should not appear as input simultaneously on the same frame
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=self.num_frames,
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )

        # optionally offload the output to CPU memory to save GPU space
        maskmem_features = current_out["maskmem_features"]
        if maskmem_features is not None:
            maskmem_features = maskmem_features.to(torch.bfloat16)
            maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
        pred_masks_gpu = current_out["pred_masks"]  # (B, 1, H, W)
        # potentially fill holes in the predicted masks
        if self.fill_hole_area > 0:
            pred_masks_gpu = fill_holes_in_mask_scores(
                pred_masks_gpu, self.fill_hole_area
            )
        pred_masks = pred_masks_gpu
        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        maskmem_pos_enc = self._get_maskmem_pos_enc(inference_state, current_out)
        # object pointer is a small tensor, so we always keep it on GPU memory for fast access
        obj_ptr = current_out["obj_ptr"]
        object_score_logits = current_out["object_score_logits"]
        best_iou_score = current_out["best_iou_score"]
        best_kf_score = current_out["kf_ious"]
        # make a compact version of this frame's output to reduce the state size
        compact_current_out = {
            "maskmem_features": maskmem_features,  # (B, C, H, W)
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": pred_masks,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
            "best_iou_score": best_iou_score,
            "kf_score": best_kf_score,
        }
        return compact_current_out, pred_masks_gpu


def recognize_from_video(predictor: SAM2VideoPredictor):
    input = args.input
    images, video_height, video_width = load_video_frames(
        video_path=input,
        # image_size=self.image_size,
    )
    images = np.load("images.npy")

    predictor.init_state(images, video_height, video_width)

    bbox = (702, 227, 1126, 938)

    _, _, masks = predictor.add_new_points_or_box(box=bbox, frame_idx=0, obj_id=0)

    for frame_idx, object_ids, masks in predictor.propagate_in_video():
        mask_to_vis = {}
        bbox_to_vis = {}

    logger.info("Script finished successfully.")


def main():
    # model files check and download
    check_and_download_models(WEIGHT_BACKBONE_PATH, MODEL_BACKBONE_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_SAM_PATH, MODEL_SAM_PATH, REMOTE_PATH)
    check_and_download_models(WEIGHT_ENC_PATH, MODEL_ENC_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        backbone = ailia.Net(MODEL_BACKBONE_PATH, WEIGHT_BACKBONE_PATH, env_id=env_id)
        sam_heads = ailia.Net(MODEL_SAM_PATH, WEIGHT_SAM_PATH, env_id=env_id)
        memory_encoder = ailia.Net(MODEL_ENC_PATH, WEIGHT_ENC_PATH, env_id=env_id)
    else:
        import onnxruntime

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        providers_cpu = ["CPUExecutionProvider"]
        backbone = onnxruntime.InferenceSession(
            WEIGHT_BACKBONE_PATH, providers=providers_cpu
        )
        sam_heads = onnxruntime.InferenceSession(WEIGHT_SAM_PATH, providers=providers)
        memory_encoder = onnxruntime.InferenceSession(
            WEIGHT_ENC_PATH, providers=providers
        )

    predictor = SAM2VideoPredictor(
        backbone=backbone,
        sam_heads=sam_heads,
        memory_encoder=memory_encoder,
        flg_onnx=args.onnx,
    )

    recognize_from_video(predictor)


if __name__ == "__main__":
    main()
