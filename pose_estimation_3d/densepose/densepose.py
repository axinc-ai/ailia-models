import os
import sys
import glob
import argparse

from typing import Any, ClassVar, Dict, List
import pickle

import cv2
from dataclasses import asdict

import model
from model import detector_postprocess
from util1 import Boxes,Instances

import torch
import numpy as np

from detectron2.config import  get_cfg, CfgNode
#from config import CfgNode, get_cfg
#from chart import DensePoseChartPredictorOutput

from util import (
    DensePoseChartPredictorOutput,
    
    DensePoseResultsContourVisualizer,
    DensePoseResultsFineSegmentationVisualizer,
    DensePoseResultsUVisualizer,
    DensePoseResultsVVisualizer,

    DensePoseResultsVisualizerWithTexture,
    get_texture_atlas,

    ScoredBoundingBoxVisualizer,

    CompoundExtractor,
    DensePoseOutputsExtractor,
    DensePoseResultExtractor,
    create_extractor,

    DensePoseOutputsTextureVisualizer,
    DensePoseOutputsVertexVisualizer,
    get_texture_atlases,

    decorate_predictor_output_class_with_confidences,
    add_densepose_config,CompoundVisualizer
)


INPUT_IMAGE = "input.jpg"
CFG ="configs/densepose_rcnn_R_50_FPN_s1x.yaml"
MODEL ="R_50_FPN_s1x.pkl"

DOC = """Apply Net - a tool to print / visualize DensePose results"""


_ACTION_REGISTRY: Dict[str, "Action"] = {}


class Action:
    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        parser.add_argument(
            "-v",
            "--verbosity",
            action="count",
            help="Verbose mode. Multiple -v options increase the verbosity.",
        )

def register_action(cls: type):
    """
    Decorator for action classes to automate action registration
    """
    global _ACTION_REGISTRY
    _ACTION_REGISTRY[cls.COMMAND] = cls
    return cls


class InferenceAction(Action):
    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        super(InferenceAction, cls).add_arguments(parser)
        #parser.add_argument("cfg", metavar="<config>" , help="Config file")
        parser.add_argument(
            "--opts",
            help="Modify config options using the command-line 'KEY VALUE' pairs",
            default=[],
            nargs=argparse.REMAINDER,
        )

    @classmethod
    def execute(cls: type, args: argparse.Namespace):
        opts = []
        cfg = cls.setup_config(CFG, MODEL, args, opts)

        #predictor = DefaultPredictor(cfg)

        file_list = cls._get_input_file_list(INPUT_IMAGE)
        if len(file_list) == 0:
            print(f"No input images for {args.input}")
            return
        context = cls.create_context(args, cfg)
        for file_name in file_list:
            img = cv2.imread(file_name)

            #outputs = predictor(img)
            #image,instances ,image_sizes, result2 = predictor(img)
            #a = model.DefaultPredictor(cfg)
            #image,instances ,image_sizes, result2 = a(img)
            #with open('data.pkl', 'wb') as f:
            #    pickle.dump((image,instances ,image_sizes, result2), f)
            with open('data.pkl', 'rb') as f:
                image,instances ,image_sizes, result2 = pickle.load(f)
            
            coarse_segm,fine_segm,u,v = result2

            def _create_output_instance( base_predictor_outputs):
                PredictorOutput = decorate_predictor_output_class_with_confidences(
                    type(base_predictor_outputs)  # pyre-ignore[6]
                )
                output = PredictorOutput(
                    **base_predictor_outputs.__dict__,
                )
                return output
            tmp = DensePoseChartPredictorOutput(coarse_segm=torch.tensor(coarse_segm),
                                                fine_segm  =torch.tensor(fine_segm),
                                                u          =torch.tensor(u),
                                                v          =torch.tensor(v))
            #tmp = DensePoseChartPredictorOutput(coarse_segm=coarse_segm,
            #                                    fine_segm  =fine_segm,
            #                                    u          =u,
            #                                    v          =v)


            densepose_predictor_outputs = _create_output_instance(tmp)

            results = instances 
            result = Instances(image_sizes[0])
            result.pred_boxes = Boxes(results[0][0][0])
            result.scores = results[0][0][1]
            pred_instances = [result]
            instances = pred_instances

            k = 0
            for detection_i in instances:
                if densepose_predictor_outputs is None:
                    continue
                n_i = detection_i.__len__()

                PredictorOutput = densepose_predictor_outputs.__class__
                #print(densepose_predictor_outputs.__class__)
                #print(PredictorOutput)

                output_i_dict = {}
                for field in asdict(densepose_predictor_outputs).keys():
                    field_value = asdict(densepose_predictor_outputs)[field]
                    # slice tensors
                    if isinstance(field_value, torch.Tensor):
                        output_i_dict[field] = field_value[k : k + n_i]
                    # leave others as is
                    else:
                        output_i_dict[field] = field_value
                detection_i.pred_densepose = PredictorOutput(**output_i_dict)
                k += n_i
            results =  instances

            height, width = img.shape[:2]
            batched_inputs = [{"image": image, "height": height, "width": width}]

            outputs = model.GeneralizedRCNN._postprocess(results, batched_inputs, image_sizes)[0]

            outputs.pred_boxes.tensor = torch.tensor(outputs.pred_boxes.tensor)
            #outputs.pred_boxes.tensor = outputs.pred_boxes.tensor

            cls.execute_on_outputs(context, {"file_name": file_name, "image": img}, outputs)
        #cls.postexecute(context)

    @classmethod
    def setup_config(
        cls: type, config_fpath: str, model_fpath: str, args: argparse.Namespace, opts: List[str]
    ):
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(config_fpath)
        cfg.merge_from_list(args.opts)
        if opts:
            cfg.merge_from_list(opts)
        cfg.MODEL.WEIGHTS = model_fpath
        cfg.freeze()
        return cfg

    @classmethod
    def _get_input_file_list(cls: type, input_spec: str):
        if os.path.isdir(input_spec):
            file_list = [
                os.path.join(input_spec, fname)
                for fname in os.listdir(input_spec)
                if os.path.isfile(os.path.join(input_spec, fname))
            ]
        elif os.path.isfile(input_spec):
            file_list = [input_spec]
        else:
            file_list = glob.glob(input_spec)
        return file_list



@register_action
class ShowAction(InferenceAction):
    """
    Show action that visualizes selected entries on an image
    """

    COMMAND: ClassVar[str] = "show"
    VISUALIZERS: ClassVar[Dict[str, object]] = {
        "dp_contour": DensePoseResultsContourVisualizer,
        "dp_segm": DensePoseResultsFineSegmentationVisualizer,
        "dp_u": DensePoseResultsUVisualizer,
        "dp_v": DensePoseResultsVVisualizer,
        "dp_iuv_texture": DensePoseResultsVisualizerWithTexture,
        "dp_cse_texture": DensePoseOutputsTextureVisualizer,
        "dp_vertex": DensePoseOutputsVertexVisualizer,
        "bbox": ScoredBoundingBoxVisualizer,
    }

    @classmethod
    def add_parser(cls: type, subparsers: argparse._SubParsersAction):
        parser = subparsers.add_parser(cls.COMMAND, help="Visualize selected entries")
        cls.add_arguments(parser)
        parser.set_defaults(func=cls.execute)

    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        super(ShowAction, cls).add_arguments(parser)
        parser.add_argument(
            "visualizations",
            metavar="<visualizations>",
            help="Comma separated list of visualizations, possible values: "
            "[{}]".format(",".join(sorted(cls.VISUALIZERS.keys()))),
        )
        parser.add_argument(
            "--min_score",
            metavar="<score>",
            default=0.8,
            type=float,
            help="Minimum detection score to visualize",
        )
        parser.add_argument(
            "--nms_thresh", metavar="<threshold>", default=None, type=float, help="NMS threshold"
        )
        parser.add_argument(
            "--texture_atlas",
            metavar="<texture_atlas>",
            default=None,
            help="Texture atlas file (for IUV texture transfer)",
        )
        parser.add_argument(
            "--texture_atlases_map",
            metavar="<texture_atlases_map>",
            default=None,
            help="JSON string of a dict containing texture atlas files for each mesh",
        )
        parser.add_argument(
            "--output",
            metavar="<image_file>",
            default="outputres.png",
            help="File name to save output to",
        )

    @classmethod
    def execute_on_outputs(
        cls: type, context: Dict[str, Any], entry: Dict[str, Any], outputs: Instances
    ):

        visualizer = context["visualizer"]
        extractor = context["extractor"]
        image_fpath = entry["file_name"]
        print(f"Processing {image_fpath}")
        image = cv2.cvtColor(entry["image"], cv2.COLOR_BGR2GRAY)
        image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
        data = extractor(outputs)
        image_vis = visualizer.visualize(image, data)
        entry_idx = context["entry_idx"] + 1
        out_fname = cls._get_out_fname(entry_idx, context["out_fname"])
        out_dir = os.path.dirname(out_fname)
        if len(out_dir) > 0 and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        cv2.imwrite(out_fname, image_vis)
        print(f"Output saved to {out_fname}")
        context["entry_idx"] += 1


    @classmethod
    def _get_out_fname(cls: type, entry_idx: int, fname_base: str):
        base, ext = os.path.splitext(fname_base)
        return base + ".{0:04d}".format(entry_idx) + ext

    @classmethod
    def create_context(cls: type, args: argparse.Namespace, cfg: CfgNode) -> Dict[str, Any]:
        vis_specs = args.visualizations.split(",")
        visualizers = []
        extractors = []
        for vis_spec in vis_specs:
            texture_atlas = get_texture_atlas(args.texture_atlas)
            texture_atlases_dict = get_texture_atlases(args.texture_atlases_map)
            vis = cls.VISUALIZERS[vis_spec](
                cfg=cfg,
                texture_atlas=texture_atlas,
                texture_atlases_dict=texture_atlases_dict,
            )
            visualizers.append(vis)
            extractor = create_extractor(vis)
            extractors.append(extractor)
        visualizer = CompoundVisualizer(visualizers)
        extractor = CompoundExtractor(extractors)
        context = {
            "extractor": extractor,
            "visualizer": visualizer,
            "out_fname": args.output,
            "entry_idx": 0,
        }
        return context


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=DOC,
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=120),
    )
    parser.set_defaults(func=lambda _: parser.print_help(sys.stdout))
    subparsers = parser.add_subparsers(title="Actions")
    for _, action in _ACTION_REGISTRY.items():
        action.add_parser(subparsers)
    return parser


def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
