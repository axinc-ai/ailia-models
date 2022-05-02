import numpy as np


class MotionTracker:
    def __init__(
            self,
            init_score_thr: float = 0.8,
            init_track_id: int = 0,
            obj_score_thr: float = 0.5,
            match_score_thr: float = 0.5,
            memo_tracklet_frames: int = 10,
            memo_backdrop_frames: int = 1,
            memo_momentum: float = 0.8,
            motion_momentum: float = 0.8,
            nms_conf_thr: float = 0.5,
            nms_backdrop_iou_thr: float = 0.3,
            nms_class_iou_thr: float = 0.7,
            loc_dim: int = 7,
            with_deep_feat: bool = True,
            with_cats: bool = True,
            with_bbox_iou: bool = True,
            with_depth_ordering: bool = True,
            with_depth_uncertainty: bool = True,
            tracker_model_name: str = 'KalmanBox3DTracker',
            lstm_name: str = 'LSTM',
            track_bbox_iou: str = 'bbox',
            depth_match_metric: str = 'centriod',
            match_metric: str = 'cycle_softmax',
            match_algo: str = 'greedy'):
        assert 0 <= memo_momentum <= 1.0
        assert 0 <= motion_momentum <= 1.0
        assert memo_tracklet_frames >= 0
        assert memo_backdrop_frames >= 0
        assert track_bbox_iou in ['box2d', 'box2d_depth_aware', 'bev', 'box3d']
        assert depth_match_metric in [
            'centroid', 'cosine', 'pure_motion', 'motion'
        ]
        assert match_metric in ['cycle_softmax', 'softmax', 'cosine']

        self.init_score_thr = init_score_thr
        self.obj_score_thr = obj_score_thr
        self.match_score_thr = match_score_thr
        self.memo_tracklet_frames = memo_tracklet_frames
        self.memo_backdrop_frames = memo_backdrop_frames
        self.memo_momentum = memo_momentum
        self.motion_momentum = motion_momentum
        self.nms_conf_thr = nms_conf_thr
        self.nms_backdrop_iou_thr = nms_backdrop_iou_thr
        self.nms_class_iou_thr = nms_class_iou_thr
        self.with_deep_feat = with_deep_feat
        self.with_cats = with_cats
        self.with_depth_ordering = with_depth_ordering
        self.with_depth_uncertainty = with_depth_uncertainty
        self.with_bbox_iou = with_bbox_iou
        self.track_bbox_iou = track_bbox_iou
        self.depth_match_metric = depth_match_metric
        # self.tracker_model = get_tracker(tracker_model_name)
        self.tracker_model_name = tracker_model_name
        self.loc_dim = loc_dim
        self.match_metric = match_metric
        self.match_algo = match_algo
        self.bbox_affinity_weight = 0.5 if with_deep_feat else 1.0
        self.feat_affinity_weight = 1 - self.bbox_affinity_weight if with_bbox_iou else 1.0
        self.num_tracklets = init_track_id
        self.tracklets = dict()
        self.backdrops = []

        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    def match(
            self,
            bboxes,
            labels,
            boxes_3d,
            depth_uncertainty,
            position,
            rotation,
            embeds,
            cur_frame: int,
            pure_det: bool = False):
        """Match incoming detection results with embedding and 3D infos

        Args:
            bboxes (torch.tensor): (N, 5), [x1, y1, x2, y2, conf]
            labels (torch.tensor): (N,)
            boxes_3d (torch.tensor): (N, 7), 3D information stored
                                     in world coordinates with the format
                                     [X, Y, Z, theta, h, w, l]
            depth_uncertainty (torch.tensor): (N, ), confidence in depth
                                     estimation
            position (torch.tensor): (3, ), camera position
            rotation (torch.tensor): (3, 3), camera rotation
            embeds (torch.tensor): (N, C), extracted box feature
            cur_frame (int): indicates the frame index
            pure_det (bool): output pure detection. Defaults False.

        Raises:
            NotImplementedError: raise if self.match_metric not found

        Returns:
            list: A list of matched bbox, labels, boxes_3d and embeds
        """
        if depth_uncertainty is None or not self.with_depth_uncertainty:
            depth_uncertainty = np.array((boxes_3d.shape[0], 1))

        _, inds = (
                bboxes[:, -1] * depth_uncertainty.flatten()
        ).sort(descending=True)
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        embeds = embeds[inds, :]
        boxes_3d = boxes_3d[inds]
        depth_uncertainty = depth_uncertainty[inds]

        if pure_det:
            valids = np.ones((bboxes.size(0)), dtype=np.bool)
            ids = np.arange(
                self.num_tracklets,
                self.num_tracklets + bboxes.size(0),
                dtype=np.long)
            self.num_tracklets += bboxes.size(0)

            return bboxes, labels, boxes_3d, ids, inds, valids

        # duplicate removal for potential backdrops and cross classes
        valids = bboxes.new_ones((bboxes.size(0)))
        ious = bbox_overlaps(bboxes[:, :-1], bboxes[:, :-1])
        for i in range(1, bboxes.size(0)):
            thr = self.nms_backdrop_iou_thr if bboxes[
                                                   i, -1] < self.obj_score_thr else self.nms_class_iou_thr
            if (ious[i, :i] > thr).any():
                valids[i] = 0
        valids = valids == 1
        bboxes = bboxes[valids, :]
        labels = labels[valids]
        embeds = embeds[valids, :]
        boxes_3d = boxes_3d[valids]
        depth_uncertainty = depth_uncertainty[valids]

        # init ids container
        ids = torch.full((bboxes.size(0),), -1, dtype=torch.long)

        # match if buffer is not empty
        if bboxes.size(0) > 0 and not self.empty:
            memo_bboxes, memo_labels, memo_boxes_3d, \
            memo_trackers, memo_embeds, memo_ids, memo_vs = self.memo

            mmcv.check_accum_time('predict', counting=True)
            memo_boxes_3d_predict = memo_boxes_3d.detach().clone()
            # print("predict---->")
            for ind, memo_tracker in enumerate(memo_trackers):
                memo_velo = memo_tracker.predict(
                    update_state=memo_tracker.age != 0)
                memo_boxes_3d_predict[ind, :3] += memo_boxes_3d.new_tensor(
                    memo_velo[7:])
            mmcv.check_accum_time('predict', counting=False)

            if self.with_bbox_iou:

                def get_xy_box(boxes_3d_world):
                    box_x_cen = boxes_3d_world[:, 0]
                    box_y_cen = boxes_3d_world[:, 1]
                    box_width = boxes_3d_world[:, 5]
                    box_length = boxes_3d_world[:, 6]

                    dets_xy_box = torch.stack([
                        box_x_cen - box_width / 2.0, box_y_cen -
                        box_length / 2.0, box_x_cen + box_width / 2.0,
                        box_y_cen + box_length / 2.0
                    ],
                        dim=1)
                    return dets_xy_box

                if self.track_bbox_iou == 'box2d':
                    scores_iou = bbox_overlaps(bboxes[:, :-1],
                                               memo_bboxes[:, :-1])
                elif self.track_bbox_iou == 'bev':
                    dets_xy_box = get_xy_box(boxes_3d)
                    memo_dets_xy_box = get_xy_box(memo_boxes_3d_predict)
                    scores_iou = bbox_overlaps(dets_xy_box, memo_dets_xy_box)
                elif self.track_bbox_iou == 'box3d':
                    depth_weight = F.pairwise_distance(
                        boxes_3d[..., None],
                        memo_boxes_3d_predict[..., None].transpose(2, 0))
                    scores_iou = torch.exp(-depth_weight / 10.0)
                elif self.track_bbox_iou == 'box2d_depth_aware':
                    depth_weight = F.pairwise_distance(
                        boxes_3d[..., None],
                        memo_boxes_3d_predict[..., None].transpose(2, 0))
                    scores_iou = torch.exp(-depth_weight / 10.0)
                    scores_iou *= bbox_overlaps(bboxes[:, :-1],
                                                memo_bboxes[:, :-1])
                else:
                    raise NotImplementedError
            else:
                scores_iou = bboxes.new_ones(
                    [bboxes.size(0), memo_bboxes.size(0)])

            if self.with_deep_feat:

                def compute_quasi_dense_feat_match(embeds, memo_embeds):
                    if self.match_metric == 'cycle_softmax':
                        feats = torch.mm(embeds, memo_embeds.t())
                        d2t_scores = feats.softmax(dim=1)
                        t2d_scores = feats.softmax(dim=0)
                        scores_feat = (d2t_scores + t2d_scores) / 2
                    elif self.match_metric == 'softmax':
                        feats = torch.mm(embeds, memo_embeds.t())
                        scores_feat = feats.softmax(dim=1)
                    elif self.match_metric == 'cosine':
                        scores_feat = torch.mm(
                            F.normalize(embeds, p=2, dim=1),
                            F.normalize(memo_embeds, p=2, dim=1).t())
                    else:
                        raise NotImplementedError
                    return scores_feat

                scores_feat = compute_quasi_dense_feat_match(
                    embeds, memo_embeds)
            else:
                scores_feat = scores_iou.new_ones(scores_iou.shape)

            # Match with depth ordering
            if self.with_depth_ordering:

                def compute_boxoverlap_with_depth(obsv_boxes_3d, memo_boxes_3d,
                                                  memo_vs):
                    # Sum up all the available region of each tracker
                    if self.depth_match_metric == 'centroid':
                        depth_weight = F.pairwise_distance(
                            obsv_boxes_3d[..., :3, None],
                            memo_boxes_3d[..., :3, None].transpose(2, 0))
                        depth_weight = torch.exp(-depth_weight / 10.0)
                    elif self.depth_match_metric == 'cosine':
                        match_corners_observe = tu.worldtocamera_torch(
                            obsv_boxes_3d[:, :3], position, rotation)
                        match_corners_predict = tu.worldtocamera_torch(
                            memo_boxes_3d[:, :3], position, rotation)
                        depth_weight = F.cosine_similarity(
                            match_corners_observe[..., None],
                            match_corners_predict[..., None].transpose(2, 0))
                        depth_weight += 1.0
                        depth_weight /= 2.0
                    elif self.depth_match_metric == 'pure_motion':
                        # Moving distance should be aligned
                        # V_observed-tracked vs. V_velocity
                        depth_weight = F.pairwise_distance(
                            obsv_boxes_3d[..., :3, None] -
                            memo_boxes_3d[..., :3, None].transpose(2, 0),
                            memo_vs[..., :3, None].transpose(2, 0))
                        depth_weight = torch.exp(-depth_weight / 5.0)
                        # Moving direction should be aligned
                        # Set to 0.5 when two vector not within +-90 degree
                        cos_sim = F.cosine_similarity(
                            obsv_boxes_3d[..., :2, None] -
                            memo_boxes_3d[..., :2, None].transpose(2, 0),
                            memo_vs[..., :2, None].transpose(2, 0))
                        cos_sim += 1.0
                        cos_sim /= 2.0
                        depth_weight *= cos_sim
                    elif self.depth_match_metric == 'motion':
                        centroid_weight = F.pairwise_distance(
                            obsv_boxes_3d[..., :3, None],
                            memo_boxes_3d_predict[..., :3,
                            None].transpose(2, 0))
                        centroid_weight = torch.exp(-centroid_weight / 10.0)
                        # Moving distance should be aligned
                        # V_observed-tracked vs. V_velocity
                        motion_weight = F.pairwise_distance(
                            obsv_boxes_3d[..., :3, None] -
                            memo_boxes_3d[..., :3, None].transpose(2, 0),
                            memo_vs[..., :3, None].transpose(2, 0))
                        motion_weight = torch.exp(-motion_weight / 5.0)
                        # Moving direction should be aligned
                        # Set to 0.5 when two vector not within +-90 degree
                        cos_sim = F.cosine_similarity(
                            obsv_boxes_3d[..., :2, None] -
                            memo_boxes_3d[..., :2, None].transpose(2, 0),
                            memo_vs[..., :2, None].transpose(2, 0))
                        cos_sim += 1.0
                        cos_sim /= 2.0
                        depth_weight = cos_sim * centroid_weight + (
                                1.0 - cos_sim) * motion_weight
                    else:
                        raise NotImplementedError

                    return depth_weight

                if self.depth_match_metric == 'motion':
                    scores_depth = compute_boxoverlap_with_depth(
                        boxes_3d, memo_boxes_3d, memo_vs)
                else:
                    scores_depth = compute_boxoverlap_with_depth(
                        boxes_3d, memo_boxes_3d_predict, memo_vs)
            else:
                scores_depth = scores_iou.new_ones(scores_iou.shape)

            if self.with_cats:
                cat_same = labels.view(-1, 1) == memo_labels.view(1, -1)
                scores_cats = cat_same.float()
            else:
                scores_cats = scores_iou.new_ones(scores_iou.shape)

            scores = self.bbox_affinity_weight * scores_iou * scores_depth + \
                     self.feat_affinity_weight * scores_feat
            scores /= (self.bbox_affinity_weight + self.feat_affinity_weight)
            scores *= (scores_iou > 0.0).float()
            scores *= (scores_depth > 0.0).float()
            scores *= scores_cats

            # Assign matching
            if self.match_algo == 'greedy':
                for i in range(bboxes.size(0)):
                    conf, memo_ind = torch.max(scores[i, :], dim=0)
                    tid = memo_ids[memo_ind]
                    # Matching confidence
                    if conf > self.match_score_thr:
                        # Update existing tracklet
                        if tid > -1:
                            # Keep object with high 3D objectness
                            if bboxes[i, -1] * depth_uncertainty[
                                i] > self.obj_score_thr:
                                ids[i] = tid
                                scores[:i, memo_ind] = 0
                                scores[i + 1:, memo_ind] = 0
                            else:
                                # Reduce FP w/ low objectness but high match conf
                                if conf > self.nms_conf_thr:
                                    ids[i] = -2
            elif self.match_algo == 'hungarian':
                # Hungarian
                matched_indices = linear_assignment(-scores.cpu().numpy())
                for idx in range(len(matched_indices[0])):
                    i = matched_indices[0][idx]
                    memo_ind = matched_indices[1][idx]
                    conf = scores[i, memo_ind]
                    tid = memo_ids[memo_ind]
                    if conf > self.match_score_thr and tid > -1:
                        # Keep object with high 3D objectness
                        if bboxes[i, -1] * depth_uncertainty[
                            i] > self.obj_score_thr:
                            ids[i] = tid
                            scores[:i, memo_ind] = 0
                            scores[i + 1:, memo_ind] = 0
                        else:
                            # Reduce FP w/ low objectness but high match conf
                            if conf > self.nms_conf_thr:
                                ids[i] = -2
                del matched_indices

        new_inds = (ids == -1) & (bboxes[:, 4] > self.init_score_thr).cpu()
        num_news = new_inds.sum()
        ids[new_inds] = torch.arange(
            self.num_tracklets,
            self.num_tracklets + num_news,
            dtype=torch.long)
        self.num_tracklets += num_news

        self.update_memo(ids, bboxes, boxes_3d, depth_uncertainty, embeds,
                         labels, cur_frame)

        update_bboxes = bboxes.detach().clone()
        update_labels = labels.detach().clone()
        update_boxes_3d = boxes_3d.detach().clone()
        for tid in ids[ids > -1]:
            update_boxes_3d[ids == tid] = self.tracklets[int(tid)]['box_3d']
        update_ids = ids.detach().clone()

        if self._debug:
            print(
                f"Updt: {update_boxes_3d.shape}\tUpdt ID: {update_ids.cpu().numpy()}\n"
                f"{update_boxes_3d.cpu().numpy()}")

        return update_bboxes, update_labels, update_boxes_3d, update_ids, inds, valids


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4)
        bboxes2 (Tensor): shape (n, 4), if is_aligned is ``True``, then m and n
            must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
    """

    assert mode in ['iou', 'iof']

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if is_aligned:
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1
    else:
        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1[:, None] + area2 - overlap)
        else:
            ious = overlap / (area1[:, None])

    return ious
