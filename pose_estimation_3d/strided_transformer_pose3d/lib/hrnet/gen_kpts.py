import argparse

from lib.hrnet.lib.utils.utilitys import PreProcess
from lib.hrnet.lib.config import cfg, update_config
from lib.hrnet.lib.utils.transforms import *
from lib.hrnet.lib.utils.inference import get_final_preds

# Loading human detector model
from lib.yolo.human_detector import yolo_human_det as yolo_det
from lib.sort import Sort

cfg_dir = 'lib/hrnet/experiments/'
model_dir = 'lib/checkpoint/'


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, default=cfg_dir + 'w48_384x288_adam_lr1e-3.yaml',
                        help='experiment configure file name')
    parser.add_argument('opts', nargs=argparse.REMAINDER, default=None,
                        help="Modify config options using the command-line")
    parser.add_argument('--modelDir', type=str, default=model_dir + 'pose_hrnet_w48_384x288.pth',
                        help='The model directory')
    parser.add_argument('--det-dim', type=int, default=416,
                        help='The input dimension of the detected image')
    parser.add_argument('--thred-score', type=float, default=0.30,
                        help='The threshold of object Confidence')
    parser.add_argument('-a', '--animation', action='store_true',
                        help='output animation')
    parser.add_argument('-np', '--num-person', type=int, default=1,
                        help='The maximum number of estimated poses')
    parser.add_argument("-v", "--video", type=str, default='camera',
                        help="input video file name")
    parser.add_argument('--gpu', type=str, default='0', help='input video')
    args = parser.parse_args()

    return args


def gen_video_kpts(frame, human_model, pose_model, det_dim=416, num_peroson=1):
    # Updating configuration
    args = parse_args()
    update_config(cfg, args)

    # Loading detector and pose model, initialize sort for track
    people_sort = Sort(min_hits=0)

    kpts_result = []
    scores_result = []
    bboxs, scores = yolo_det(frame, human_model, reso=det_dim)

    # Using Sort to track people
    people_track = people_sort.update(bboxs)

    # Track the first two people in the video and remove the ID
    if people_track.shape[0] == 1:
        people_track_ = people_track[-1, :-1].reshape(1, 4)
    elif people_track.shape[0] >= 2:
        people_track_ = people_track[-num_peroson:, :-1].reshape(num_peroson, 4)
        people_track_ = people_track_[::-1]
    else:
        return None, None

    track_bboxs = []
    for bbox in people_track_:
        bbox = [round(i, 2) for i in list(bbox)]
        track_bboxs.append(bbox)
    # bbox is coordinate location
    inputs, origin_img, center, scale = PreProcess(frame, track_bboxs, cfg, num_peroson)

    inputs = inputs[:, [2, 1, 0]]
    output = pose_model.run(inputs.astype('float32'))[0]

    # compute coordinate
    preds, maxvals = get_final_preds(cfg,  output, np.asarray(center), np.asarray(scale))

    kpts = np.zeros((num_peroson, 17, 2), dtype=np.float32)
    scores = np.zeros((num_peroson, 17), dtype=np.float32)
    for i, kpt in enumerate(preds):
        kpts[i] = kpt

    for i, score in enumerate(maxvals):
        scores[i] = score.squeeze()

    kpts_result.append(kpts)
    scores_result.append(scores)

    keypoints = np.array(kpts_result)
    scores = np.array(scores_result)

    keypoints = keypoints.transpose(1, 0, 2, 3)  # (T, M, N, 2) --> (M, T, N, 2)
    scores = scores.transpose(1, 0, 2)  # (T, M, N) --> (M, T, N)

    return keypoints, scores
