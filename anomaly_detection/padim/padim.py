import os
import sys
import time
from collections import OrderedDict
import random
import pickle
from typing import Tuple, List, Dict, Union, Optional, Any

import numpy as np
import cv2
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

import ailia

# import original modules
sys.path.append('../../util')
from arg_utils import get_base_parser, update_parser, get_savepath
from model_utils import check_and_download_models
from detector_utils import load_image
import webcamera_utils

# logger
from logging import getLogger

from padim_utils import *

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/padim/'

IMAGE_PATH = './bottle_000.png'
SAVE_IMAGE_PATH = './output.png'
IMAGE_RESIZE = 480
IMAGE_SIZE = 448
KEEP_ASPECT = True

# ======================
# Argument Parser Config
# ======================

def create_parser():
    """Create and configure the argument parser."""
    parser = get_base_parser('PaDiM model', IMAGE_PATH, SAVE_IMAGE_PATH)
    parser.add_argument(
        '-a', '--arch', default='resnet18', choices=('resnet18', 'wide_resnet50_2'),
        help='Architecture model.'
    )
    parser.add_argument(
        '-f', '--feat', metavar="FILE", default=None,
        help='Train set feature files.'
    )
    parser.add_argument(
        '-bs', '--batch_size_training', type=int, default=4,
        help='Batch size.'
    )
    parser.add_argument(
        '-bst', '--batch_size_testing', type=int, default=2,
        help='Batch size.'
    )
    parser.add_argument(
        '-tr', '--train_dir', metavar="DIR", default="./train",
        help='Directory of the train files.'
    )
    parser.add_argument(
        '-gt', '--gt_dir', metavar="DIR", default="./gt_masks",
        help='Directory of the ground truth mask files.'
    )
    parser.add_argument(
        '--seed', type=int, default=1024,
        help='Random seed'
    )
    parser.add_argument(
        '-th', '--threshold', type=float, default=None,
        help='Threshold'
    )
    parser.add_argument(
        '-ag', '--aug', action='store_true',
        help='Process with augmentation.'
    )
    parser.add_argument(
        '-an', '--aug_num', type=int, default=5,
        help='Specify the amplification number of augmentation.'
    )
    parser.add_argument(
        '-eon', '--enable_optimization', action='store_true',
        help='Flag to enable optimized code'
    )
    parser.add_argument(
        '--save_format', metavar="FORMAT", default="pkl", choices=("pkl", "npy", "pt"),
        help='Choose training file format: pt, npy or pkl.'
    )
    parser.add_argument(
        '--optimization_device', metavar="device", default='cpu', choices=('cpu', 'cuda', 'mps'),
        help='Choose optimization device'
    )
    
    return parser

# ======================
# Helper functions
# ======================

def setup_optimization(enable_optimization,  optimization_device):
    """Setup optimization environment if enabled."""
    if enable_optimization:
        import torch
        if optimization_device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        elif optimization_device == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        weights_torch = gaussian_kernel1d_torch(4, 0, int(4.0 * float(4) + 0.5), device).unsqueeze(0).unsqueeze(0).expand(1, 1, 33)
        logger.info(f"Torch device: {device}")
        return device, weights_torch
    return None, None

def visualize_results(file_list, test_imgs, scores, anormal_scores, gt_imgs, threshold, savepath, enable_optimization):
    """
    Visualize and save the anomaly detection results.
    
    Args:
        file_list: List of input image paths
        test_imgs: Test images
        scores: Anomaly scores
        anormal_scores: Global anomaly scores
        gt_imgs: Ground truth images
        threshold: Threshold value
        savepath: Path to save visualization results
    """
    num = len(file_list)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    
    for i in range(num):
        image_path = file_list[i]
        img = test_imgs[i]
        img = denormalization(img)
        
        if gt_imgs is not None and i < len(gt_imgs) and gt_imgs[i] is not None:
            gt = gt_imgs[i]
            gt = gt.transpose(1, 2, 0).squeeze()
        else:
            gt = np.zeros((1, 1, 1))
            
        # Get visualization based on optimization flag
        if enable_optimization:
            heat_map, mask, vis_img = visualize(img, scores[i].squeeze(0).cpu().numpy(), threshold)
        else:
            heat_map, mask, vis_img = visualize(img, scores[i], threshold)

        # Create the visualization
        fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        fig_img.suptitle(f"Input: {image_path}  Anomaly score: {anormal_scores[i]:.4f}")
        logger.info(f"Anomaly score: {anormal_scores[i]:.4f}")

        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
            
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        
        ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[2].imshow(img, cmap='gray', interpolation='none')
        ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[2].title.set_text('Predicted heat map')
        
        ax_img[3].imshow(mask, cmap='gray')
        ax_img[3].title.set_text('Predicted mask')
        
        ax_img[4].imshow(vis_img)
        ax_img[4].title.set_text('Segmentation result')
        
        # Add colorbar
        left, bottom, width, height = 0.92, 0.15, 0.015, 0.7
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        # Save the figure
        if '.' in savepath.split('/')[-1]:
            savepath_tmp = get_savepath(savepath, image_path, ext='.png')
        else:
            filename_tmp = image_path.split('/')[-1]
            ext_tmp = '.' + filename_tmp.split('.')[-1]
            filename_tmp = filename_tmp.replace(ext_tmp, '.png')
            savepath_tmp = f'{savepath}/{filename_tmp}'
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(savepath_tmp), exist_ok=True)
        logger.info(f'Saved at: {savepath_tmp}')
        fig_img.savefig(savepath_tmp, dpi=100)
        plt.close()

def infer_benchmark(net, params, train_outputs, img, image_size, enable_optimization, device=None, weights_torch=None, benchmark_count=10):
    """Run inference in benchmark mode and measure performance."""
    total_time = 0
    
    if enable_optimization:
        for i in range(benchmark_count):
            start = int(round(time.time() * 1000))
            dist_tmp = infer_optimized(net, params, train_outputs, img, image_size, device, logger, weights_torch)
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end - start} ms')
            if i != 0:  # Skip first run (warmup)
                total_time += (end - start)
    else:
        for i in range(benchmark_count):
            start = int(round(time.time() * 1000))
            dist_tmp = infer(net, params, train_outputs, img, image_size)
            end = int(round(time.time() * 1000))
            logger.info(f'\tailia processing time {end - start} ms')
            if i != 0:  # Skip first run (warmup)
                total_time += (end - start)
                
    avg_time = total_time / (benchmark_count - 1)
    logger.info(f'\taverage time {avg_time:.2f} ms')
    return dist_tmp

def infer_init_run(net, params, train_outputs, image_size, enable_optimization, device=None, weights_torch=None):
    """Perform initialization inference with dummy data."""
    logger.info("PaDiM initialization inference starts!")
    
    # Create dummy image (batch size 1)
    dummy_image = np.random.rand(1, 3, image_size, image_size).astype(np.float32) * 255.0
    
    if enable_optimization:
        score = infer_optimized(net, params, train_outputs, dummy_image, image_size, device, logger, weights_torch, args.batch_size_testing)
    else:
        score = infer(net, params, train_outputs, dummy_image, image_size)
        
    logger.info("PaDiM initialization inference finished!")

def load_ground_truth_images(input_paths, gt_dir, image_resize, image_size, keep_aspect=True):
    """Load ground truth images for evaluation."""
    gt_imgs = []
    
    if not gt_dir or not os.path.exists(gt_dir):
        return [None] * len(input_paths)
    
    for image_path in input_paths:
        fname = os.path.splitext(os.path.basename(image_path))[0]
        gt_fpath = os.path.join(gt_dir, fname + '_mask.png')
        
        if os.path.exists(gt_fpath):
            gt_img = load_image(gt_fpath)
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGRA2RGB)
            gt_img = preprocess(gt_img, image_resize, mask=True, keep_aspect=keep_aspect, crop_size=image_size)
            
            if gt_img is not None:
                gt_img = gt_img[0, [0]]
            else:
                gt_img = np.zeros((1, image_size, image_size))
        else:
            gt_img = None
            
        gt_imgs.append(gt_img)
        
    return gt_imgs

# ======================
# Training and inference functions
# ======================

def train_model(net, params, args, enable_optimization, device=None, weights_torch=None):
    """
    Train the model or load pre-trained features.
    
    Args:
        net: Neural network model
        params: Model parameters
        args: Command line arguments
        device: Torch device for optimization
        weights_torch: Pre-computed torch weights for optimization
        
    Returns:
        Train outputs for inference
    """
    if args.feat and os.path.exists(args.feat):
        train_outputs = load_training_file(args.feat, args.save_format, enable_optimization, device)
        logger.info('Loaded pre-trained features.')
    else:
        logger.info('Training model from images...')
        train_outputs = train_from_images(net, params, args, enable_optimization, device, weights_torch)
        
    return train_outputs

def train_from_images(net, params, args, enable_optimization, device=None, weights_torch=None):
    """Train the model using images from the training directory."""
    
    if enable_optimization:
        train_outputs = training_optimized(
            net, params, args.IMAGE_RESIZE, args.IMAGE_SIZE, args.KEEP_ASPECT, 
            args.batch_size_training, args.train_dir, args.aug, args.aug_num, args.seed, logger
        )

    else:
        train_outputs = training(
            net, params, args.IMAGE_RESIZE, args.IMAGE_SIZE, args.KEEP_ASPECT, 
            args.batch_size_training, args.train_dir, args.aug, args.aug_num, args.seed, logger
        )
    
    # Save training outputs
    train_outputs = save_training_file(args.feat or os.path.basename(args.train_dir), args.save_format, 
                      train_outputs, enable_optimization, device)
    print("train_outputs type ", type(train_outputs[0]), type(train_outputs[1]), type(train_outputs[2]))
    return train_outputs

def save_training_file(train_feat_file, save_format, train_outputs, enable_optimization, device=None):
    """Save training features to a file."""
    filename_base = train_feat_file.split(".")[0].strip() if train_feat_file else "train"
    
    if enable_optimization:
        # Convert numpy arrays to torch tensors for optimized version
        if not isinstance(train_outputs[0], torch.Tensor):
            train_outputs_torch = [
                torch.from_numpy(train_outputs[0]).float().to(device), 
                train_outputs[1],
                torch.from_numpy(train_outputs[2]).float().to(device), 
                train_outputs[3]
            ]
        else:
            train_outputs_torch = train_outputs
            
        if save_format == "npy":
            for i, output in enumerate(train_outputs):
                output_filename = f"{filename_base}_{i}.npy"
                logger.info(f'Saving train feature to: {output_filename}')
                # Convert tensor to numpy if needed
                if isinstance(output, torch.Tensor):
                    np.save(output_filename, output.cpu().numpy())
                else:
                    np.save(output_filename, output)
                
        elif save_format == "pkl":
            output_filename = f"{filename_base}_optimized.pkl"
            logger.info(f'Saving train feature to: {output_filename}')
            with open(output_filename, 'wb') as f:
                pickle.dump(train_outputs_torch, f)
                
        elif save_format == "pt":
            output_filename = f"{filename_base}_optimized.pt"
            logger.info(f'Saving train feature to: {output_filename}')
            torch.save(train_outputs_torch, output_filename)
            
        return train_outputs_torch
    else:
        # Non-optimized version
        if save_format == "npy":
            for i, output in enumerate(train_outputs):
                output_filename = f"{filename_base}_{i}.npy"
                logger.info(f'Saving train feature to: {output_filename}')
                np.save(output_filename, output)
                
        elif save_format == "pkl":
            output_filename = f"{filename_base}.pkl"
            logger.info(f'Saving train feature to: {output_filename}')
            with open(output_filename, 'wb') as f:
                pickle.dump(train_outputs, f)
                
        elif save_format == "pt":
            output_filename = f"{filename_base}.pt"
            logger.info(f'Saving train feature to: {output_filename}')
            torch.save(train_outputs, output_filename)
            
    logger.info('Features saved.')
    return train_outputs

def load_training_file(train_feat_file, save_format, enable_optimization, device=None):
    """Load training features from a file."""
    if not train_feat_file:
        if save_format == "npy":
            base_filename = "train"
        elif enable_optimization:
            base_filename = f"train_optimized.{save_format}"
        else:
            base_filename = f"train.{save_format}"
    else:
        base_filename = train_feat_file
        
    if not os.path.exists(base_filename) and save_format != "npy":
        raise FileNotFoundError(f"Training file {base_filename} not found")
    
    try:
        if enable_optimization:
            if save_format == "pkl":
                logger.info(f"Loading {base_filename}")
                with open(base_filename, 'rb') as f:
                    train_outputs = pickle.load(f)
                # Move tensors to device
                return [
                    train_outputs[0].to(device), 
                    train_outputs[1],
                    train_outputs[2].to(device), 
                    train_outputs[3]
                ]
                
            elif save_format == "npy":
                train_outputs = []
                i = 0
                filename_base = base_filename.split("_")[0].strip()
                
                # Load all numbered npy files
                while True:
                    try:
                        npy_file = f"{filename_base}_{i}.npy"
                        logger.info(f"Loading {npy_file}")
                        train_outputs.append(np.load(npy_file, allow_pickle=True))
                        i += 1
                    except FileNotFoundError:
                        if i == 0:
                            raise FileNotFoundError(f"No training files found with pattern {filename_base}_*.npy")
                        break
                
                # Convert to torch tensors
                return [
                    torch.from_numpy(train_outputs[0]).float().to(device),
                    train_outputs[1],
                    torch.from_numpy(train_outputs[2]).float().to(device),
                    train_outputs[3]
                ]
                
            elif save_format == "pt":
                logger.info(f"Loading {base_filename}")
                return torch.load(base_filename, map_location=device)
                
        else:
            # Non-optimized version
            if save_format == "pkl":
                logger.info(f"Loading {base_filename}")
                with open(base_filename, 'rb') as f:
                    return pickle.load(f)
                    
            elif save_format == "npy":
                train_outputs = []
                i = 0
                filename_base = base_filename.split("_")[0].strip()
                
                while True:
                    try:
                        npy_file = f"{filename_base}_{i}.npy"
                        logger.info(f"Loading {npy_file}")
                        train_outputs.append(np.load(npy_file, allow_pickle=True))
                        i += 1
                    except FileNotFoundError:
                        if i == 0:
                            raise FileNotFoundError(f"No training files found with pattern {filename_base}_*.npy")
                        break
                        
                return train_outputs
                
            elif save_format == "pt":
                logger.info(f"Loading {base_filename}")
                train_outputs = torch.load(base_filename)
                
                # Convert tensors to numpy if needed
                if isinstance(train_outputs[0], torch.Tensor):
                    return [item.cpu().numpy() if isinstance(item, torch.Tensor) else item 
                           for item in train_outputs]
                return train_outputs
                
    except Exception as e:
        logger.error(f"Error loading training file: {e}")
        raise

def determine_threshold(net, params, train_outputs, input_paths, gt_imgs, args, enable_optimization, device=None, weights_torch=None):
    """Determine optimal threshold based on ground truth images."""
    if args.threshold is not None:
        return args.threshold
        
    if args.video:
        logger.info('Please set threshold manually for video mode')
        return 0.5
        
    if not gt_imgs or all(gt is None for gt in gt_imgs):
        logger.warning("No ground truth images found. Using default threshold of 0.5")
        return 0.5
    
    logger.info("Determining optimal threshold...")
    score_maps = []
    

    for i, image_path in enumerate(input_paths):
        logger.info(f'Processing ({image_path})')
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img = preprocess(img, args.IMAGE_RESIZE, keep_aspect=args.KEEP_ASPECT, crop_size=args.IMAGE_SIZE)

        if enable_optimization:
            print("train_outputs type: ", type(train_outputs[0]))
            dist_tmp = infer_optimized(net, params, train_outputs, img, args.IMAGE_SIZE, device, logger, weights_torch)
        else:
            dist_tmp = infer(net, params, train_outputs, img, args.IMAGE_SIZE)
            
        score_maps.append(dist_tmp)

    # Calculate scores and threshold
    if enable_optimization:
        scores = normalize_scores_torch(score_maps, args.IMAGE_SIZE)
        threshold = decide_threshold(scores.cpu().numpy(), gt_imgs)
    else:
        scores = normalize_scores(score_maps, args.IMAGE_SIZE)
        threshold = decide_threshold(scores, gt_imgs)
    
    logger.info(f'Optimal threshold: {threshold}')
    return threshold

def infer_images(net, params, train_outputs, threshold, gt_imgs, args, enable_optimization, device=None, weights_torch=None):
    """Process images using the trained model and visualize results."""
    if not args.input:
        logger.error("Input file not found")
        return
        
    test_imgs = []
    score_maps = []
        
    # Run initialization
    infer_init_run(net, params, train_outputs, args.IMAGE_SIZE, enable_optimization, device, weights_torch)
    
    for i, image_path in enumerate(args.input):
        logger.info(f'Processing ({image_path})')
        
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img = preprocess(img, args.IMAGE_RESIZE, keep_aspect=args.KEEP_ASPECT, crop_size=args.IMAGE_SIZE)
        
        test_imgs.append(img[0])
        
        if args.benchmark:
            dist_tmp = infer_benchmark(net, params, train_outputs, img, args.IMAGE_SIZE, enable_optimization,
                                       device, weights_torch, args.benchmark_count)
        else:
            if enable_optimization:
                dist_tmp = infer_optimized(net, params, train_outputs, img, args.IMAGE_SIZE,  device, logger, weights_torch)
            else:
                dist_tmp = infer(net, params, train_outputs, img, args.IMAGE_SIZE)
                
        if dist_tmp is not None:
            score_maps.append(dist_tmp)
    
    # Calculate scores
    if enable_optimization:
        scores = normalize_scores_torch(score_maps, args.IMAGE_SIZE)
        anormal_scores = calculate_anormal_scores_torch(score_maps, args.IMAGE_SIZE)
    else:
        scores = normalize_scores(score_maps, args.IMAGE_SIZE)
        anormal_scores = calculate_anormal_scores(score_maps, args.IMAGE_SIZE)
    
    # Visualize results
    visualize_results(args.input, test_imgs, scores, anormal_scores, gt_imgs, threshold, args.savepath, enable_optimization)

def infer_video(net, params, train_outputs, threshold, args, enable_optimization, device=None, weights_torch=None):
    """Process video using the trained model and visualize results in real-time."""
    capture = webcamera_utils.get_capture(args.video)
    f_h = args.IMAGE_SIZE
    f_w = int(args.IMAGE_SIZE * 2.8)
    # Setup video writer if needed
    if args.savepath == SAVE_IMAGE_PATH:
        args.savepath="result.mp4"
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    
    score_map = []
    frame_shown = False
    
    # Run initialization
    infer_init_run(net, params, train_outputs, args.IMAGE_SIZE, enable_optimization, device, weights_torch)
    
    try:
        while True:
            ret, frame = capture.read()
            if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
                break
            if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
                break
            
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = preprocess(img, args.IMAGE_RESIZE, keep_aspect=args.KEEP_ASPECT, crop_size=args.IMAGE_SIZE)
            
            if enable_optimization:
                dist_tmp = infer_optimized(net, params, train_outputs, img, args.IMAGE_SIZE, device, logger, weights_torch, args.batch_size_testing)
                norm_scores_tensor = normalize_scores_torch([dist_tmp], args.IMAGE_SIZE)
                score_value = norm_scores_tensor[0].cpu().numpy()[0]
            else:
                dist_tmp = infer(net, params, train_outputs, img, args.IMAGE_SIZE)
                norm_scores = normalize_scores([dist_tmp], args.IMAGE_SIZE, None)
                score_value = norm_scores[0]
            
            # Visualize the result: denormalize, create heat map, mask, and final visualization.
            denorm_img = denormalization(img[0])
            heat_map, mask, vis_img = visualize(denorm_img, score_value, threshold)
            frame_out = pack_visualize(heat_map, mask, vis_img, np.asarray([score_value]), args.IMAGE_SIZE)
            
            # Ensure the frame has the exact dimensions expected by the writer.
            frame_out = cv2.resize(frame_out, (f_w, f_h))            
            cv2.imshow('frame', frame_out)
            
            if writer is not None:
                writer.write(frame_out)

    finally:
        capture.release()
        cv2.destroyAllWindows()
        if writer is not None:
            writer.release()
# ======================
# Main functions
# ======================

def main():
    """Main function for PaDiM anomaly detection."""
    start_time = time.time()
    
    # Parse arguments
    parser = create_parser()
    global args
    args = update_parser(parser)
    
    # Setup global variables for common use
    args.IMAGE_RESIZE = IMAGE_RESIZE
    args.IMAGE_SIZE = IMAGE_SIZE
    args.KEEP_ASPECT = KEEP_ASPECT
    
    # Setup optimization if enabled
    device, weights_torch = setup_optimization(args.enable_optimization,  args.optimization_device)
    print(device)
    # Get model parameters
    weight_path, model_path, params = get_params(args.arch)
    
    # Check and download models if needed
    check_and_download_models(weight_path, model_path, REMOTE_PATH)
    # Create neural network
    net = ailia.Net(model_path, weight_path, env_id=args.env_id)
    
    
    # Train or load the model
    train_outputs = train_model(net, params, args, args.enable_optimization, device, weights_torch)
    # Process video or images
    if args.video:
        if not args.threshold:
            args.threshold=0.5
        infer_video(net, params, train_outputs, args.threshold, args, args.enable_optimization, device, weights_torch)
    else:
        # Load ground truth images if available
        gt_imgs = load_ground_truth_images(args.input, args.gt_dir, args.IMAGE_RESIZE, args.IMAGE_SIZE, args.KEEP_ASPECT)
        
        # Determine threshold
        if not args.threshold:
            args.threshold = determine_threshold(net, params, train_outputs, args.input, gt_imgs, args, args.enable_optimization, device, weights_torch)
        infer_images(net, params, train_outputs, args.threshold, gt_imgs, args, args.enable_optimization, device, weights_torch)
        
    execution_time = int((time.time() - start_time) * 1000)
    logger.info(f'Script finished successfully. Execution time: {execution_time}ms')

if __name__ == '__main__':
    main()