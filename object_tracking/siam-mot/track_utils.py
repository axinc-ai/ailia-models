import numpy as np

from this_utils import boxes_nms, boxes_filter


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

    # @staticmethod
    # def swap_pairs(entity_list):
    #     assert len(entity_list) % 2 == 0
    #     # Take the targets of the other frame (in a tracking pair) as input during training, thus swap order
    #     for xx in range(0, len(entity_list), 2):
    #         entity_list[xx], entity_list[xx + 1] = entity_list[xx + 1], entity_list[xx]
    #     return entity_list
    #
    # @staticmethod
    # def shuffle_feature(f):
    #     """
    #     odd-even order swap of the feature tensor in the batch dimension
    #     """
    #
    #     def shuffle_feature_tensor(x):
    #         batch_size = x.shape[0]
    #         assert batch_size % 2 == 0
    #
    #         # get channel swap order [1, 0, 3, 2, ...]
    #         odd_idx = range(1, batch_size, 2)
    #         even_idx = range(0, batch_size, 2)
    #         idxs = np.arange(0, batch_size)
    #         idxs[even_idx] = idxs[even_idx] + 1
    #         idxs[odd_idx] = idxs[odd_idx] - 1
    #         idxs = torch.tensor(idxs)
    #
    #         return x[idxs]
    #
    #     if isinstance(f, tuple):
    #         shuffle_f = []
    #         for i, _f in enumerate(f):
    #             shuffle_f.append(shuffle_feature_tensor(_f))
    #         shuffle_f = tuple(shuffle_f)
    #     else:
    #         shuffle_f = shuffle_feature_tensor(f)
    #
    #     return shuffle_f
    #
    # def extend_bbox(self, in_box: [BoxList]):
    #     """
    #     Extend the bounding box to define the search region
    #     :param in_box: a set of bounding boxes in previous frame
    #     :param min_wh: the miniumun width/height of the search region
    #     """
    #     for i, _track in enumerate(in_box):
    #         bbox_w = _track.bbox[:, 2] - _track.bbox[:, 0] + 1
    #         bbox_h = _track.bbox[:, 3] - _track.bbox[:, 1] + 1
    #         w_ext = bbox_w * (self.search_expansion / 2.)
    #         h_ext = bbox_h * (self.search_expansion / 2.)
    #
    #         # todo: need to check the equation later
    #         min_w_ext = (self.min_search_wh - bbox_w) / (self.search_expansion * 2.)
    #         min_h_ext = (self.min_search_wh - bbox_h) / (self.search_expansion * 2.)
    #
    #         w_ext = torch.max(min_w_ext, w_ext)
    #         h_ext = torch.max(min_h_ext, h_ext)
    #         in_box[i].bbox[:, 0] -= w_ext
    #         in_box[i].bbox[:, 1] -= h_ext
    #         in_box[i].bbox[:, 2] += w_ext
    #         in_box[i].bbox[:, 3] += h_ext
    #         # in_box[i].clip_to_image()
    #     return in_box
    #
    # def pad_feature(self, f):
    #     """
    #     Pad the feature maps with 0
    #     :param f: [torch.tensor] or torch.tensor
    #     """
    #
    #     if isinstance(f, (list, tuple)):
    #         pad_f = []
    #         for i, _f in enumerate(f):
    #             # todo fix this hack, should read from cfg file
    #             pad_pixels = int(self.pad_pixels / ((2 ** i) * 4))
    #             x = F.pad(_f, [pad_pixels, pad_pixels, pad_pixels, pad_pixels],
    #                       mode='constant', value=0)
    #             pad_f.append(x)
    #         pad_f = tuple(pad_f)
    #     else:
    #         pad_f = F.pad(f, [self.pad_pixels, self.pad_pixels,
    #                           self.pad_pixels, self.pad_pixels],
    #                       mode='constant', value=0)
    #
    #     return pad_f
    #
    # def update_boxes_in_pad_images(self, boxlists: [BoxList]):
    #     """
    #     Update the coordinates of bounding boxes in the padded image
    #     """
    #
    #     pad_width = self.pad_pixels
    #     pad_height = self.pad_pixels
    #
    #     pad_boxes = []
    #     for _boxlist in boxlists:
    #         im_width, im_height = _boxlist.size
    #         new_width = int(im_width + pad_width * 2)
    #         new_height = int(im_height + pad_height * 2)
    #
    #         assert (_boxlist.mode == 'xyxy')
    #         xmin, ymin, xmax, ymax = _boxlist.bbox.split(1, dim=-1)
    #         new_xmin = xmin + pad_width
    #         new_ymin = ymin + pad_height
    #         new_xmax = xmax + pad_width
    #         new_ymax = ymax + pad_height
    #         bbox = torch.cat((new_xmin, new_ymin, new_xmax, new_ymax), dim=-1)
    #         bbox = BoxList(bbox, [new_width, new_height], mode='xyxy')
    #         for _field in _boxlist.fields():
    #             bbox.add_field(_field, _boxlist.get_field(_field))
    #         pad_boxes.append(bbox)
    #
    #     return pad_boxes


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
        print("active_tracks---", active_tracks.bbox)

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
        features = torch.cat(buffer_feat + cached_features)
        sr = cat_boxlist(track_memory[1] + [x[1] for x in dormant_caches])
        boxes = cat_boxlist(track_memory[2] + [x[2] for x in dormant_caches])
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
        start_idxs = ((_ids < 0) & (_scores >= self.start_thresh)).nonzero()

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


track_utils = TrackUtils(
    search_expansion=1.0,
    min_search_wh=0,
    pad_pixels=512)

track_pool = TrackPool(max_dormant_frames=1)

track_head = TrackHead(track_utils, track_pool)

track_solver = TrackSolver(track_pool)
