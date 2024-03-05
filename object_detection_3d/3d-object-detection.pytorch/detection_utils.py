from collections import namedtuple
import queue

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

from objectron.dataset import graphics

__all__ = [
    'draw_kp',
    'IOUTracker',
]

TrackedObj = namedtuple('TrackedObj', 'rect kp label')


class Track:
    def __init__(self, ID, bbox, kps, time, align_kp=False):
        self.id = ID
        self.boxes = [bbox]
        self.kps = [kps]
        self.timestamps = [time]
        self.no_updated_frames = 0
        self.align_kp = align_kp

    def get_end_time(self):
        return self.timestamps[-1]

    def get_start_time(self):
        return self.timestamps[0]

    def get_last_box(self):
        return self.boxes[-1]

    def get_last_kp(self):
        return self.kps[-1]

    def __len__(self):
        return len(self.timestamps)

    def _interpolate(self, target_box, target_kp, timestamp, skip_size):
        last_box = self.get_last_box()
        last_kp = self.get_last_kp()
        for t in range(1, skip_size):
            interp_box = [int(b1 + (b2 - b1) / skip_size * t) for b1, b2 in zip(last_box, target_box)]
            interp_kp = [k1 + (k2 - k1) / skip_size * t for k1, k2 in zip(last_kp, target_kp)]
            self.boxes.append(interp_box)
            self.kps.append(interp_kp)
            self.timestamps.append(self.get_end_time() + 1)

    def _filter_last_3d_box(self, filter_speed, add_treshold, no_updated_frames_treshold):
        if self.timestamps[-1] - self.timestamps[-2] == 1:
            num_keypoints = len(self.kps[-2]) // 2
            self.kps[-2] = np.array(self.kps[-2]).reshape(num_keypoints, 2)
            self.kps[-1] = self.kps[-1].reshape(num_keypoints, 2)
            # compute average distance before run
            add_dist = np.mean(np.linalg.norm(self.kps[-1] - self.kps[-2], axis=1))
            if self.align_kp:
                indexes_to_revert = self._align_kp_positions()
                rearranged_kps = self.kps[-1][indexes_to_revert]
                add_dist_after = np.mean(np.linalg.norm(rearranged_kps - self.kps[-2], axis=1))
                if add_dist_after < add_dist:
                    considered_kps = rearranged_kps
                    add_dist = add_dist_after
                else:
                    considered_kps = self.kps[-1]
            else:
                considered_kps = self.kps[-1]
            # if add distance is appropriate for previous frame by given treshold
            # then we smooth kps with EMA
            if add_dist < add_treshold:
                self.no_updated_frames = 0
                filtered_kps = (1 - filter_speed) * self.kps[-2] + filter_speed * considered_kps
            elif self.no_updated_frames > no_updated_frames_treshold:
                # if bbox haven't been updated too long -> interrupt EMA
                # and get new bbox
                filtered_kps = considered_kps
            else:
                # if not -> use bbox from previous frame
                filtered_kps = self.kps[-2]
                self.no_updated_frames += 1

            self.kps[-1] = tuple(filtered_kps.reshape(-1).tolist())

    def _align_kp_positions(self):
        # store indexes for matching
        num_keypoints = self.kps[-1].shape[0]
        indexes = list(range(num_keypoints))
        # list for marking vertexes
        ind_updated = [False] * num_keypoints
        for i in range(len(self.kps[-1])):
            if ind_updated[i]:
                continue
            distance = np.linalg.norm(self.kps[-1][i, :] - self.kps[-2][i, :])
            min_d_idx = i
            for j in range(i + 1, len(self.kps[-1])):
                d = np.linalg.norm(self.kps[-1][i, :] - self.kps[-2][j, :])
                if d < distance:
                    min_d_idx = j
            # if we already rearranged vertexes we will not do it twice to prevent
            # indexes mess
            if min_d_idx != i and not ind_updated[i] and not ind_updated[min_d_idx]:
                # swap vertexes
                indexes[i] = min_d_idx
                indexes[min_d_idx] = i
                # mark vertexes as visited
                ind_updated[i] = True
                ind_updated[min_d_idx] = True

        return indexes

    def _filter_last_box(self, filter_speed):
        if self.timestamps[-1] - self.timestamps[-2] == 1:
            filtered_box = list(self.boxes[-2])
            for j in range(len(self.boxes[-1])):
                filtered_box[j] = int((1 - filter_speed) * filtered_box[j]
                                      + filter_speed * self.boxes[-1][j])
            self.boxes[-1] = tuple(filtered_box)

    def add_detection(self, bbox, kps, timestamp, max_skip_size=1,
                      box_filter_speed=0.7, kp_filter_speed=0.3,
                      add_treshold=0.1, no_updated_frames_treshold=5):
        skip_size = timestamp - self.get_end_time()
        if 1 < skip_size <= max_skip_size:
            self._interpolate(bbox, kps, timestamp, skip_size)
            assert self.get_end_time() == timestamp - 1

        self.boxes.append(bbox)
        self.kps.append(kps)
        self.timestamps.append(timestamp)
        self._filter_last_box(box_filter_speed)
        self._filter_last_3d_box(kp_filter_speed, add_treshold, no_updated_frames_treshold)


class IOUTracker:
    def __init__(
            self,
            time_window=5,
            continue_time_thresh=2,
            track_clear_thresh=3000,
            match_threshold=0.4,
            track_detection_iou_thresh=0.5,
            interpolate_time_thresh=10,
            detection_filter_speed=0.7,
            keypoints_filter_speed=0.3,
            add_treshold=0.1,
            no_updated_frames_treshold=5,
            align_kp=False):

        self.last_global_id = 0
        self.global_ids_queue = queue.Queue()
        self.tracks = []
        self.history_tracks = []
        self.time = 0
        assert time_window >= 1
        self.time_window = time_window
        assert continue_time_thresh >= 1
        self.continue_time_thresh = continue_time_thresh
        assert track_clear_thresh >= 1
        self.track_clear_thresh = track_clear_thresh
        assert 0 <= match_threshold <= 1
        self.match_threshold = match_threshold
        assert 0 <= track_detection_iou_thresh <= 1
        self.track_detection_iou_thresh = track_detection_iou_thresh
        assert interpolate_time_thresh >= 0
        self.interpolate_time_thresh = interpolate_time_thresh
        assert 0 <= detection_filter_speed <= 1
        self.detection_filter_speed = detection_filter_speed
        assert 0 <= keypoints_filter_speed <= 1
        self.keypoints_filter_speed = keypoints_filter_speed
        assert 0 <= add_treshold <= 1
        self.add_treshold = add_treshold
        assert no_updated_frames_treshold >= 0
        assert isinstance(no_updated_frames_treshold, int)
        self.align_kp = align_kp
        self.no_updated_frames_treshold = no_updated_frames_treshold
        self.current_detections = None

    def process(self, frame, detections, kps):
        assignment = self._continue_tracks(detections, kps)
        self._create_new_tracks(detections, kps, assignment)
        self._clear_old_tracks()
        self.time += 1

    def get_tracked_objects(self):
        label = 'ID'
        objs = []
        for track in self.tracks:
            if track.get_end_time() == self.time - 1 and len(track) > self.time_window:
                objs.append(TrackedObj(track.get_last_box(), track.get_last_kp(),
                                       label + ' ' + str(track.id)))
            elif track.get_end_time() == self.time - 1 and len(track) <= self.time_window:
                objs.append(TrackedObj(track.get_last_box(), track.get_last_kp(), label + ' -1'))
        return objs

    def get_tracks(self):
        return self.tracks

    def get_archived_tracks(self):
        return self.history_tracks

    def _continue_tracks(self, detections, kps):
        active_tracks_idx = []
        for i, track in enumerate(self.tracks):
            if track.get_end_time() >= self.time - self.continue_time_thresh:
                active_tracks_idx.append(i)

        cost_matrix = self._compute_detections_assignment_cost(active_tracks_idx, detections)

        assignment = [None for _ in range(cost_matrix.shape[0])]
        if cost_matrix.size > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for i, j in zip(row_ind, col_ind):
                idx = active_tracks_idx[j]
                if cost_matrix[i, j] < self.match_threshold and \
                        self._iou(self.tracks[idx].boxes[-1], detections[i]) > self.track_detection_iou_thresh:
                    assignment[i] = j

            for i, j in enumerate(assignment):
                if j is not None:
                    idx = active_tracks_idx[j]
                    self.tracks[idx].add_detection(detections[i], kps[i],
                                                   self.time, self.continue_time_thresh,
                                                   self.detection_filter_speed, self.keypoints_filter_speed,
                                                   self.add_treshold, self.no_updated_frames_treshold)
        return assignment

    def _clear_old_tracks(self):
        clear_tracks = []
        for track in self.tracks:
            # remove too old tracks
            if track.get_end_time() < self.time - self.track_clear_thresh:
                self.history_tracks.append(track)
                continue
            # remove too short and outdated tracks
            if track.get_end_time() < self.time - self.continue_time_thresh \
                    and len(track) < self.time_window:
                self.global_id_releaser(track.id)
                continue
            clear_tracks.append(track)
        self.tracks = clear_tracks

    def _compute_detections_assignment_cost(self, active_tracks_idx, detections):
        cost_matrix = np.zeros((len(detections), len(active_tracks_idx)), dtype=np.float32)

        for i, idx in enumerate(active_tracks_idx):
            track_box = self.tracks[idx].get_last_box()
            for j, d in enumerate(detections):
                iou_dist = 0.5 * (1 - self._giou(d, track_box))
                cost_matrix[j, i] = iou_dist

        return cost_matrix

    def _create_new_tracks(self, detections, kps, assignment):
        for i, j in enumerate(assignment):
            if j is None:
                self.tracks.append(Track(self.global_id_getter(),
                                         detections[i], kps[i], self.time, self.align_kp))

    def global_id_getter(self):
        if self.global_ids_queue.empty():
            self.global_ids_queue.put(self.last_global_id)
            self.last_global_id += 1

        return self.global_ids_queue.get_nowait()

    def global_id_releaser(self, ID):
        assert ID <= self.last_global_id
        self.global_ids_queue.put(ID)

    @staticmethod
    def _area(bbox):
        return max((bbox[2] - bbox[0]), 0) * max((bbox[3] - bbox[1]), 0)

    def _giou(self, b1, b2, a1=None, a2=None):
        if a1 is None:
            a1 = self._area(b1)
        if a2 is None:
            a2 = self._area(b2)
        intersection = self._area([max(b1[0], b2[0]), max(b1[1], b2[1]),
                                   min(b1[2], b2[2]), min(b1[3], b2[3])])

        enclosing = self._area([min(b1[0], b2[0]), min(b1[1], b2[1]),
                                max(b1[2], b2[2]), max(b1[3], b2[3])])
        u = a1 + a2 - intersection
        iou = intersection / u if u > 0 else 0
        giou = iou - (enclosing - u) / enclosing if enclosing > 0 else -1
        return giou

    def _iou(self, b1, b2, a1=None, a2=None):
        if a1 is None:
            a1 = self._area(b1)
        if a2 is None:
            a2 = self._area(b2)
        intersection = self._area([max(b1[0], b2[0]), max(b1[1], b2[1]),
                                   min(b1[2], b2[2]), min(b1[3], b2[3])])

        u = a1 + a2 - intersection
        return intersection / u if u > 0 else 0


def normalize(image_shape, unnormalized_keypoints):
    ''' normalize keypoints to image coordinates '''
    assert len(image_shape) in [2, 3]
    if len(image_shape) == 3:
        h, w, _ = image_shape
    else:
        h, w = image_shape

    keypoints = unnormalized_keypoints / np.asarray([w, h], np.float32)
    return keypoints


def draw_kp(
        img, keypoints, normalized=True, num_keypoints=9, label=None):
    '''
    img: numpy three dimensional array
    keypoints: array like with shape [9,2]
    name: path to save
    '''
    img_copy = img.copy()
    # if image transposed
    if img_copy.shape[0] == 3:
        img_copy = np.transpose(img_copy, (1, 2, 0))
    # expand dim with zeros, needed for drawing function API
    expanded_kp = np.zeros((num_keypoints, 3))
    keypoints = keypoints if normalized else normalize(img_copy.shape, keypoints)
    expanded_kp[:, :2] = keypoints
    graphics.draw_annotation_on_image(img_copy, expanded_kp, [num_keypoints])
    # put class label if given
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_copy, str(label), (10, 180), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return img_copy
