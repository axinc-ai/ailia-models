import numpy as np

from math_utils import softmax

from this_utils import BBox, sigmoid, boxes_nms, boxes_filter, boxes_cat


class TrackUtils(object):
    """
    A class that includes utility functions unique to track branch
    """

    def __init__(self, search_expansion=1.0, min_search_wh=128, pad_pixels=256):
        """
        :param search_expansion: expansion ratio (of the search region)
        w.r.t the size of tracking targets
        :param min_search_wh: minimal size (width and height) of the search region
        :param pad_pixels: the padding pixels that are neccessary to keep the
        feature map pf search region and that of template target in the same scale
        """
        self.search_expansion = search_expansion
        self.min_search_wh = min_search_wh
        self.pad_pixels = pad_pixels

    def extend_bbox(self, in_box):
        """
        Extend the bounding box to define the search region
        """
        for i, _track in enumerate(in_box):
            bbox_w = _track.bbox[:, 2] - _track.bbox[:, 0] + 1
            bbox_h = _track.bbox[:, 3] - _track.bbox[:, 1] + 1
            w_ext = bbox_w * (self.search_expansion / 2.)
            h_ext = bbox_h * (self.search_expansion / 2.)

            # todo: need to check the equation later
            min_w_ext = (self.min_search_wh - bbox_w) / (self.search_expansion * 2.)
            min_h_ext = (self.min_search_wh - bbox_h) / (self.search_expansion * 2.)

            # w_ext = np.max(min_w_ext, w_ext)
            w_ext = np.max(np.vstack([min_w_ext, w_ext]), axis=0)
            # h_ext = np.max(min_h_ext, h_ext)
            h_ext = np.max(np.vstack([min_h_ext, h_ext]), axis=0)
            in_box[i].bbox[:, 0] -= w_ext
            in_box[i].bbox[:, 1] -= h_ext
            in_box[i].bbox[:, 2] += w_ext
            in_box[i].bbox[:, 3] += h_ext

        return in_box

    def update_boxes_in_pad_images(self, boxlists):
        """
        Update the coordinates of bounding boxes in the padded image
        """

        pad_width = self.pad_pixels
        pad_height = self.pad_pixels

        pad_boxes = []
        for _boxlist in boxlists:
            # im_width, im_height = _boxlist.size
            # new_width = int(im_width + pad_width * 2)
            # new_height = int(im_height + pad_height * 2)

            # xmin, ymin, xmax, ymax = _boxlist.bbox.split(1, dim=-1)
            xmin, ymin, xmax, ymax = np.split(_boxlist.bbox, 4, axis=1)
            new_xmin = xmin + pad_width
            new_ymin = ymin + pad_height
            new_xmax = xmax + pad_width
            new_ymax = ymax + pad_height
            bbox = np.concatenate((new_xmin, new_ymin, new_xmax, new_ymax), axis=1)
            bbox = BBox(
                bbox=bbox,
                scores=_boxlist.scores,
                ids=_boxlist.ids,
                labels=_boxlist.labels,
            )
            pad_boxes.append(bbox)

        return pad_boxes


class TrackPool(object):
    """
    A class to manage the track id distribution (initiate/kill a track)
    """

    def __init__(self, active_ids=None, max_entangle_length=10, max_dormant_frames=1):
        if active_ids is None:
            self._active_ids = set()
            # track ids that are killed up to previous frames
            self._dormant_ids = {}
            # track ids that are killed in current frame
            self._kill_ids = set()
            self._max_id = -1
        self._embedding = None
        self._cache = {}
        self._frame_idx = 0
        self._max_dormant_frames = max_dormant_frames
        self._max_entangle_length = max_entangle_length

    def expire_tracks(self):
        """
        Expire the suspended tracks after they are inactive
        for a consecutive self._max_dormant_frames frames
        """
        for track_id, last_active in list(self._dormant_ids.items()):
            if self._frame_idx - last_active >= self._max_dormant_frames:
                self._dormant_ids.pop(track_id)
                self._kill_ids.add(track_id)
                self._cache.pop(track_id, None)

    def increment_frame(self, value=1):
        self._frame_idx += value

    def update_cache(self, cache):
        """
        Update the latest position (bbox) / search region / template feature
        for each track in the cache
        """
        template_features, sr, template_boxes = cache
        sr = sr[0]
        template_boxes = template_boxes[0]
        for idx in range(len(template_boxes.bbox)):
            if len(template_features) > 0:
                assert len(template_features) == len(sr.bbox)
                features = template_features[idx]
            else:
                features = template_features
            search_region = BBox(
                bbox=sr.bbox[idx: idx + 1],
                scores=sr.scores[idx: idx + 1],
                ids=sr.ids[idx: idx + 1],
                labels=sr.labels[idx: idx + 1],
            )
            box = BBox(
                bbox=template_boxes.bbox[idx: idx + 1],
                scores=template_boxes.scores[idx: idx + 1],
                ids=template_boxes.ids[idx: idx + 1],
                labels=template_boxes.labels[idx: idx + 1],
            )
            track_id = box.ids[0]
            self._cache[track_id] = (features, search_region, box)

    def resume_track(self, track_id):
        """
        Resume a dormant track
        """
        if track_id not in self._dormant_ids or \
                track_id in self._active_ids:
            raise ValueError

        self._active_ids.add(track_id)
        self._dormant_ids.pop(track_id)

    def kill_track(self, track_id):
        """
        Kill a track
        """
        if track_id not in self._active_ids:
            raise ValueError

        self._active_ids.remove(track_id)
        self._kill_ids.add(track_id)
        self._cache.pop(track_id, None)

    def start_track(self):
        """
        Return a new track id, when starting a new track
        """
        new_id = self._max_id + 1
        self._max_id = new_id
        self._active_ids.add(new_id)

        return new_id

    def get_active_ids(self):
        return self._active_ids

    def get_dormant_ids(self):
        return set(self._dormant_ids.keys())

    def get_cache(self):
        return self._cache

    def activate_tracks(self, track_id):
        if track_id in self._active_ids or \
                track_id not in self._dormant_ids:
            raise ValueError

        self._active_ids.add(track_id)
        self._dormant_ids.pop(track_id)

    def reset(self):
        self._active_ids = set()
        self._kill_ids = set()
        self._dormant_ids = {}
        self._embedding = None
        self._cache = {}
        self._max_id = -1
        self._frame_idx = 0


class TrackHead(object):
    def __init__(self, track_utils, track_pool):
        super(TrackHead, self).__init__()

        self.feature_extractor = None

        self.track_utils = track_utils
        self.track_pool = track_pool

    def get_track_memory(self, features, track, extract_cache):
        active_tracks = self._get_track_targets(track)
        print("active_tracks---", active_tracks.bbox.shape)
        print("active_tracks---", active_tracks.ids)
        print("active_tracks---", active_tracks.labels)

        # no need for feature extraction of search region if
        # the tracker is tracktor, or no trackable instances
        if len(active_tracks.bbox) == 0:
            import copy
            template_features = np.array([])
            sr = copy.deepcopy(active_tracks)
            sr.size = [
                1280 + self.track_utils.pad_pixels * 2,
                800 + self.track_utils.pad_pixels * 2]
            track_memory = (template_features, [sr], [active_tracks])
        else:
            track_memory = extract_cache(features, active_tracks)
            print("track_memory------", len(track_memory))
            print("track_memory------", track_memory[0].shape)
            print("track_memory------", track_memory[1])
            print("track_memory------", track_memory[2])

        track_memory = self._update_memory_with_dormant_track(track_memory)

        self.track_pool.update_cache(track_memory)

        return track_memory

    def _update_memory_with_dormant_track(self, track_memory):
        cache = self.track_pool.get_cache()
        if not cache or track_memory is None:
            return track_memory

        dormant_caches = []
        for dormant_id in self.track_pool.get_dormant_ids():
            if dormant_id in cache:
                dormant_caches.append(cache[dormant_id])
        cached_features = [x[0][None, ...] for x in dormant_caches]
        if track_memory[0] is None:
            if track_memory[1][0] or track_memory[2][0]:
                raise Exception("Unexpected cache state")
            track_memory = [[]] * 3
            buffer_feat = []
        else:
            buffer_feat = [track_memory[0]]
        # TODO value check
        features = np.concatenate(buffer_feat + cached_features, axis=0)
        sr = boxes_cat(track_memory[1] + [x[1] for x in dormant_caches])
        boxes = boxes_cat(track_memory[2] + [x[2] for x in dormant_caches])
        return features, [sr], [boxes]

    def _get_track_targets(self, target):
        if len(target.bbox) == 0:
            return target
        active_ids = self.track_pool.get_active_ids()

        ids = target.ids.tolist()
        idxs = np.zeros((len(ids),), dtype=np.bool)
        for _i, _id in enumerate(ids):
            if _id in active_ids:
                idxs[_i] = True

        idxs = idxs.nonzero()[0]

        return boxes_filter(target, idxs)


class TrackSolver(object):
    def __init__(
            self,
            track_pool,
            track_thresh=0.3,
            start_track_thresh=0.5,
            resume_track_thresh=0.4):

        self.track_pool = track_pool
        self.track_thresh = track_thresh
        self.start_thresh = start_track_thresh
        self.resume_track_thresh = resume_track_thresh

    def get_nms_boxes(self, detection):
        detection = boxes_nms(detection, nms_thresh=0.5)

        _ids = detection.ids
        _scores = detection.scores

        _scores[_scores >= 2.] = _scores[_scores >= 2.] - 2.
        _scores[_scores >= 1.] = _scores[_scores >= 1.] - 1.

        return detection, _ids, _scores

    def solve(self, boxes):
        """
        The solver is to merge predictions from detection branch as well as from track branch.
        The goal is to assign an unique track id to bounding boxes that are deemed tracked
        :param boxes: it includes three set of distinctive prediction:
        prediction propagated from active tracks, (2 >= score > 1, id >= 0),
        prediction propagated from dormant tracks, (2 >= score > 1, id >= 0),
        prediction from detection (1 > score > 0, id = -1).
        :return:
        """

        if len(boxes.bbox) == 0:
            return boxes

        track_pool = self.track_pool

        all_ids = boxes.ids
        all_scores = boxes.scores
        active_ids = track_pool.get_active_ids()
        dormant_ids = track_pool.get_dormant_ids()

        active_mask = np.array([int(x) in active_ids for x in all_ids])

        # differentiate active tracks from dormant tracks with scores
        # active tracks, (3 >= score > 2, id >= 0),
        # dormant tracks, (2 >= score > 1, id >= 0),
        # By doing this, dormant tracks will be merged to active tracks during nms,
        # if they highly overlap
        all_scores[active_mask] += 1.
        print("all_scores------", all_scores)
        print("all_scores------", all_scores.shape)

        nms_detection, nms_ids, nms_scores = self.get_nms_boxes(boxes)

        combined_detection = nms_detection
        _ids = combined_detection.ids
        _scores = combined_detection.scores

        # start track ids
        start_idxs = ((_ids < 0) & (_scores >= self.start_thresh)).nonzero()[0]

        # inactive track ids
        inactive_idxs = ((_ids >= 0) & (_scores < self.track_thresh))
        nms_track_ids = set(_ids[_ids >= 0].tolist())
        all_track_ids = set(all_ids[all_ids >= 0].tolist())
        # active tracks that are removed by nms
        nms_removed_ids = all_track_ids - nms_track_ids
        inactive_ids = set(_ids[inactive_idxs].tolist()) | nms_removed_ids

        # resume dormant mask, if needed
        dormant_mask = np.array([int(x) in dormant_ids for x in _ids])
        resume_ids = _ids[dormant_mask & (_scores >= self.resume_track_thresh)]
        for _id in resume_ids.tolist():
            track_pool.resume_track(_id)

        for _idx in start_idxs:
            _ids[_idx] = track_pool.start_track()

        active_ids = track_pool.get_active_ids()
        for _id in inactive_ids:
            if _id in active_ids:
                track_pool.suspend_track(_id)

        # make sure that the ids for inactive tracks in current frame are meaningless (< 0)
        _ids[inactive_idxs] = -1

        track_pool.expire_tracks()
        track_pool.increment_frame()

        print("combined_detection---", combined_detection.bbox)
        print("combined_detection---", combined_detection.bbox.shape)
        return combined_detection


def decode_response(
        cls_logits, center_logits, reg_logits, locations, boxes,
        use_centerness=True, sigma=0.4):
    cls_logits = softmax(cls_logits, axis=1)
    cls_logits = cls_logits[:, 1:2, :, :]
    if use_centerness:
        centerness = sigmoid(center_logits)
        obj_confidence = cls_logits * centerness
    else:
        obj_confidence = cls_logits

    num_track_objects = obj_confidence.shape[0]
    obj_confidence = obj_confidence.reshape((num_track_objects, -1))
    tlbr = reg_logits.reshape((num_track_objects, 4, -1))

    scale_penalty = _get_scale_penalty(tlbr, boxes)
    cos_window = _get_cosine_window_penalty(tlbr)
    p_obj_confidence = (obj_confidence * scale_penalty) * (1 - sigma) + sigma * cos_window

    idxs = np.argmax(p_obj_confidence, axis=1)

    target_ids = np.arange(num_track_objects)
    bb_c = locations[target_ids, idxs, :]
    shift_tlbr = tlbr[target_ids, :, idxs]

    bb_tl_x = bb_c[:, 0:1] - shift_tlbr[:, 0:1]
    bb_tl_y = bb_c[:, 1:2] - shift_tlbr[:, 1:2]
    bb_br_x = bb_c[:, 0:1] + shift_tlbr[:, 2:3]
    bb_br_y = bb_c[:, 1:2] + shift_tlbr[:, 3:4]
    bb = np.concatenate((bb_tl_x, bb_tl_y, bb_br_x, bb_br_y), axis=1)

    cls_logits = cls_logits.reshape((num_track_objects, -1))
    bb_conf = cls_logits[target_ids, idxs]

    return bb, bb_conf


def _get_scale_penalty(tlbr: np.ndarray, boxes: BBox):
    box_w = boxes.bbox[:, 2] - boxes.bbox[:, 0]
    box_h = boxes.bbox[:, 3] - boxes.bbox[:, 1]

    r_w = tlbr[:, 2] + tlbr[:, 0]
    r_h = tlbr[:, 3] + tlbr[:, 1]

    scale_w = r_w / box_w[:, None]
    scale_h = r_h / box_h[:, None]
    scale_w = np.max(
        np.concatenate([
            scale_w[:, None, :], (1 / scale_w)[:, None, :]
        ], axis=1), axis=1)
    scale_h = np.max(
        np.concatenate([
            scale_h[:, None, :], (1 / scale_h)[:, None, :]
        ], axis=1), axis=1)

    scale_penalty = np.exp((-scale_w * scale_h + 1) * 0.1)

    return scale_penalty


def _get_cosine_window_penalty(tlbr: np.ndarray):
    num_boxes, _, num_elements = tlbr.shape
    h_w = int(np.sqrt(num_elements))
    hanning = np.hanning(h_w + 1)[:h_w]
    window = np.outer(hanning, hanning)
    window = window.reshape(-1)

    return window[None, :]


def results_to_boxes(bb, bb_conf, boxes: [BBox], amodal=False):
    num_boxes_per_image = [len(b.bbox) for b in boxes]
    bb = np.split(bb, num_boxes_per_image, axis=0)
    bb_conf = np.split(bb_conf, num_boxes_per_image, axis=0)

    track_boxes = []
    for _bb, _bb_conf, _boxes in zip(bb, bb_conf, boxes):
        _bb = _bb.reshape(-1, 4)
        track_box = BBox(
            bbox=_bb,
            ids=_boxes.ids,
            labels=_boxes.labels,
            scores=_bb_conf,
        )
        if not amodal:
            track_box.bbox[:, 0] = track_box.bbox[:, 0].clip(0, max=1280 - 1)
            track_box.bbox[:, 1] = track_box.bbox[:, 1].clip(0, max=800 - 1)
            track_box.bbox[:, 2] = track_box.bbox[:, 2].clip(0, max=1280 - 1)
            track_box.bbox[:, 3] = track_box.bbox[:, 3].clip(0, max=800 - 1)
            box = track_box.bbox
            keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
            track_box = boxes_filter(track_box, keep)

        track_boxes.append(track_box)

    return track_boxes


def get_locations(
        fmap: np.ndarray, template_fmap: np.ndarray,
        sr_boxes: [BBox], shift_xy, up_scale=1):
    """

    """
    h, w = fmap.shape[-2:]
    h, w = h * up_scale, w * up_scale
    concat_boxes = np.concatenate([b.bbox for b in sr_boxes], axis=0)
    box_w = concat_boxes[:, 2] - concat_boxes[:, 0]
    box_h = concat_boxes[:, 3] - concat_boxes[:, 1]
    stride_h = box_h / (h - 1)
    stride_w = box_w / (w - 1)

    delta_x = np.arange(0, w, dtype=np.float32)
    delta_y = np.arange(0, h, dtype=np.float32)
    delta_x = (concat_boxes[:, 0])[:, None] + delta_x[None, :] * stride_w[:, None]
    delta_y = (concat_boxes[:, 1])[:, None] + delta_y[None, :] * stride_h[:, None]

    h0, w0 = template_fmap.shape[-2:]
    assert (h0 == w0)
    border = np.int(np.floor(h0 / 2))
    st_end_idx = int(border * up_scale)
    delta_x = delta_x[:, st_end_idx:-st_end_idx]
    delta_y = delta_y[:, st_end_idx:-st_end_idx]

    locations = []
    num_boxes = delta_x.shape[0]
    for i in range(num_boxes):
        _x, _y = np.meshgrid(delta_x[i, :], delta_y[i, :])
        _y = _y.reshape(-1)
        _x = _x.reshape(-1)
        _xy = np.stack((_x, _y), axis=1)
        locations.append(_xy)
    locations = np.stack(locations)

    # shift the coordinates w.r.t the original image space (before padding)
    locations[:, :, 0] -= shift_xy[0]
    locations[:, :, 1] -= shift_xy[1]

    return locations


track_utils = TrackUtils(
    search_expansion=1.0,
    min_search_wh=0,
    pad_pixels=512)

track_pool = TrackPool(max_dormant_frames=1)

track_head = TrackHead(track_utils, track_pool)

track_solver = TrackSolver(track_pool)
