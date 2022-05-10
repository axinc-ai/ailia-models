import os

import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from matplotlib.ticker import MultipleLocator
from pyquaternion import Quaternion

from logging import getLogger  # noqa

import tracking_utils as tu

logger = getLogger(__name__)


def xywh2xyxy(bbox: list):
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]


def xywh2center(bbox: list):
    return [bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0]


def read_coco(coco_annos, category):
    anno_dict = {}
    # obj_num = 0
    # total_frame = len(coco_annos['images'])
    # total_annotations = len(coco_annos['annotations'])

    map_cat_dict = {
        cat_dict['id']: cat_dict['name'].capitalize()
        for cat_dict in coco_annos['categories']
    }

    for seq_num, log_dict in enumerate(coco_annos['videos']):
        log_id = log_dict['id']
        # print(f"{log_dict['id']} {log_name}")

        anno_dict[log_id] = {
            'frames': {},
            'seq_name': log_dict['name'],
            'width': None,
            'height': None
        }

    for fr_num, img_dict in enumerate(coco_annos['images']):
        # print(f"{fr_num}/{total_frame}")

        frm_anno_dict = {
            'cam_loc': img_dict['pose']['position'],
            'cam_rot': img_dict['pose']['rotation'],
            'cam_calib': img_dict['cali'],
            'im_path': img_dict['file_name'],
            'annotations': []
        }
        log_id = img_dict['video_id']
        fr_id = img_dict['index']
        anno_dict[log_id]['frames'][fr_id] = frm_anno_dict.copy()
        if fr_id == 0:
            anno_dict[log_id]['width'] = img_dict['width']
            anno_dict[log_id]['height'] = img_dict['height']

    for obj_num, obj_dict in enumerate(coco_annos.get('annotations', [])):
        if map_cat_dict[obj_dict['category_id']] not in category:
            continue

        log_id = coco_annos['images'][obj_dict['image_id']]['video_id']
        fr_id = coco_annos['images'][obj_dict['image_id']]['index']
        frm_anno_dict = anno_dict[log_id]['frames'][fr_id]
        # print(f"{obj_num}/{total_annotations}")
        t_data = {
            'fr_id': obj_dict['image_id'],
            'track_id': obj_dict['instance_id'],
            'obj_type': map_cat_dict[obj_dict['category_id']],
            'truncated': obj_dict['is_truncated'],
            'occluded': obj_dict['is_occluded'],
            'alpha': obj_dict['alpha'],
            'box': xywh2xyxy(obj_dict['bbox']),
            'box_center': obj_dict.get(
                'center_2d', xywh2center(obj_dict['bbox'])),
            'dimension': obj_dict['dimension'],
            'location': obj_dict['translation'],
            'yaw': obj_dict['roty'],
            'confidence': obj_dict.get('uncertainty', 0.95),
            'score': obj_dict.get('score', 1.0)
        }
        if len(frm_anno_dict['annotations']) > 0:
            frm_anno_dict['annotations'].append(t_data.copy())
        else:
            frm_anno_dict['annotations'] = [t_data.copy()]

    return anno_dict


class RandomColor:
    def __init__(self, n, name='hsv'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a
        distinct
        RGB color; the keyword argument name must be a standard mpl colormap
        name.'''
        self.cmap = plt.cm.get_cmap(name, n)
        self.n = n

    def get_random_color(self, scale=1):
        ''' Using scale = 255 for opencv while scale = 1 for matplotlib '''
        return tuple(
            [scale * x for x in self.cmap(np.random.randint(self.n))[:3]])


def fig2data(fig, size: tuple = None):
    fig.canvas.figure.tight_layout()
    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()

    # canvas.tostring_argb give pixmap in ARGB mode.
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)

    buf.shape = (h, w, 4)  # last dim: (alpha, r, g, b)

    # Roll the ALPHA channel to have it in RGBA mode
    # buf = np.roll(buf, 3, axis=2)

    # Take only RGB
    buf = buf[:, :, 1:]

    if size is not None:
        buf = cv2.resize(buf, size)

    # Get BGR from RGB
    buf = buf[:, :, ::-1]

    return buf


def plot_bev_obj(
        ax: plt.axes,
        center: np.ndarray,
        center_hist: np.ndarray,
        yaw: np.ndarray,
        yaw_hist: np.ndarray,
        l: float,
        w: float,
        color: list,
        text: str,
        line_width: int = 1):
    # Calculate length, width of object
    vec_l = [l * np.cos(yaw), -l * np.sin(yaw)]
    vec_w = [-w * np.cos(yaw - np.pi / 2), w * np.sin(yaw - np.pi / 2)]
    vec_l = np.array(vec_l)
    vec_w = np.array(vec_w)

    # Make 4 points
    p1 = center + 0.5 * vec_l - 0.5 * vec_w
    p2 = center + 0.5 * vec_l + 0.5 * vec_w
    p3 = center - 0.5 * vec_l + 0.5 * vec_w
    p4 = center - 0.5 * vec_l - 0.5 * vec_w

    # Plot object
    line_style = '-' if 'PD' in text else ':'
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
            line_style,
            c=color,
            linewidth=3 * line_width)
    ax.plot([p1[0], p4[0]], [p1[1], p4[1]],
            line_style,
            c=color,
            linewidth=line_width)
    ax.plot([p3[0], p2[0]], [p3[1], p2[1]],
            line_style,
            c=color,
            linewidth=line_width)
    ax.plot([p3[0], p4[0]], [p3[1], p4[1]],
            line_style,
            c=color,
            linewidth=line_width)

    # Plot center history
    for index, ct in enumerate(center_hist):
        yaw = yaw_hist[index].item()
        vec_l = np.array([l * np.cos(yaw), -l * np.sin(yaw)])
        ct_dir = ct + 0.5 * vec_l
        alpha = max(float(index) / len(center_hist), 0.5)
        ax.plot([ct[0], ct_dir[0]], [ct[1], ct_dir[1]],
                line_style,
                alpha=alpha,
                c=color,
                linewidth=line_width)
        ax.scatter(
            ct[0],
            ct[1],
            alpha=alpha,
            c=np.array([color]),
            linewidth=line_width)


class Pose:
    ''' Calibration matrices in KITTI
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
        X (z) ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord (KITTI):
        front x, left y, up z

        world coord (GTA):
        right x, front y, up z

        velodyne coord (nuScenes):
        right x, front y, up z

        velodyne coord (Waymo):
        front x, left y, up z

        rect/ref camera coord (KITTI, GTA, nuScenes):
        right x, down y, front z

        camera coord (Waymo):
        front x, left y, up z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf
    '''

    def __init__(self, position, rotation):
        # relative position to the 1st frame: (X, Y, Z)
        # relative rotation to the previous frame: (r_x, r_y, r_z)
        self.position = position
        if rotation.shape == (3, 3):
            # rotation matrices already
            self.rotation = rotation
        else:
            # rotation vector
            self.rotation = tu.angle2rot(np.array(rotation))


def merge_vid(vidname1, vidname2, outputname):
    print(f"Vertically stack {vidname1} and {vidname2}, save as {outputname}")
    os.makedirs(os.path.dirname(outputname), exist_ok=True)

    # Get input video capture
    cap1 = cv2.VideoCapture(vidname1)
    cap2 = cv2.VideoCapture(vidname2)

    # Default resolutions of the frame are obtained.The default resolutions
    # are system dependent.
    # We convert the resolutions from float to integer.
    # https://docs.opencv.org/2.4/modules/highgui/doc
    # /reading_and_writing_images_and_video.html#videocapture-get
    frame_width = int(cap1.get(3))
    frame_height = int(cap1.get(4))
    fps1 = cap1.get(5)
    FOURCC = int(cap1.get(6))
    num_frames = int(cap1.get(7))

    # frame_width2 = int(cap2.get(3))
    frame_height2 = int(cap2.get(4))
    fps2 = cap2.get(5)
    num_frames2 = int(cap2.get(7))

    if fps1 > fps2:
        fps = fps1
    else:
        fps = fps2

    # assert frame_height == frame_height2, \
    #     f"Height of frames are not equal. {frame_height} vs. {frame_height2}"
    assert num_frames == num_frames2, \
        f"Number of frames are not equal. {num_frames} vs. {num_frames2}"

    # Set output videowriter
    vidsize = (frame_width + frame_height, frame_height)
    out = cv2.VideoWriter(outputname, FOURCC, fps, vidsize)

    print(f"Total {num_frames} frames. Now saving...")

    # Loop over and save
    idx = 0
    while (cap1.isOpened() and cap2.isOpened() and idx < num_frames):
        ret1 = ret2 = False
        frame1 = frame2 = None
        if idx % (fps / fps1) == 0.0:
            # print(idx, fps/fps2, "1")
            ret1, frame1 = cap1.read()
        if idx % (fps / fps2) == 0.0:
            # print(idx, fps/fps1, "2")
            ret2, frame2 = cap2.read()
            if frame_height != frame_height2:
                frame2 = cv2.resize(frame2, (frame_height, frame_height))
        # print(ret1, ret2)
        if ret1 and ret2:
            out_frame = np.hstack([frame1, frame2])
            out.write(out_frame)
        idx += 1

    out.release()
    cap1.release()
    cap2.release()


class Visualizer:
    def __init__(
            self,
            res_folder: str,
            fps: float = 7.0,
            draw_bev: bool = True,
            draw_2d: bool = False,
            draw_3d: bool = True,
            draw_traj: bool = True,
            draw_tid: bool = True,
            is_merge: bool = True,
    ):
        # Parameters
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SCALE: float = 1.0
        self.FONT_THICKNESS: int = 1
        self.FOURCC = cv2.VideoWriter_fourcc(*'mp4v')
        self.FOCAL_LENGTH = None
        self.fps: float = fps

        np.random.seed(777)

        # Create canvas
        self.fig_size: int = 10
        self.dpi: int = 100
        self.bev_size: int = self.fig_size * self.dpi
        self.fig, self.ax = plt.subplots(
            figsize=(self.fig_size, self.fig_size), dpi=self.dpi)
        self.x_min: int = -55
        self.x_max: int = 55
        self.y_min: int = 0
        self.y_max: int = 100
        self.interval: int = 10
        self.num_hist: int = 10

        self.res_folder = res_folder
        self.draw_bev: bool = draw_bev
        self.draw_2d: bool = draw_2d
        self.draw_3d: bool = draw_3d
        self.draw_traj: bool = draw_traj
        self.draw_tid: bool = draw_tid
        self.is_merge: bool = is_merge

        # Variables
        self.trk_vid_name: str = None
        self.bev_vid_name: str = None

    @staticmethod
    def get_3d_info(anno, cam_calib, cam_pose):
        h, w, l = anno['dimension']
        depth = anno['location'][2]
        alpha = anno['alpha']
        xc, yc = anno['box_center']
        obj_class = anno['obj_type']

        points_cam = tu.imagetocamera(
            np.array([[xc, yc]]), np.array([depth]), cam_calib)

        bev_center = points_cam[0, [0, 2]]
        yaw = tu.alpha2rot_y(alpha, bev_center[0], bev_center[1])  # rad
        quat_yaw = Quaternion(axis=[0, 1, 0], radians=yaw)
        quat_cam_rot = Quaternion(matrix=cam_pose.rotation)
        quat_yaw_world = quat_cam_rot * quat_yaw

        box3d = tu.computeboxes([yaw], (h, w, l), points_cam)
        points_world = tu.cameratoworld(points_cam, cam_pose.position, cam_pose.rotation)

        output = {
            'center': bev_center,
            'loc_cam': points_cam,
            'loc_world': points_world,
            'yaw': yaw,
            'yaw_world_quat': quat_yaw_world,
            'box3d': box3d,
            'class': obj_class
        }
        return output

    @staticmethod
    def draw_3d_traj(
            frame,
            points_hist,
            cam_calib,
            cam_pose,
            line_color=(0, 255, 0)):
        # Plot center history
        for index, wt in enumerate(points_hist):
            ct = tu.worldtocamera(wt, cam_pose.position, cam_pose.rotation)
            pt = tu.cameratoimage(ct, cam_calib)
            rgba = line_color + tuple(
                [int(max(float(index) / len(points_hist), 0.5) * 255)])
            cv2.circle(
                frame, (int(pt[0, 0]), int(pt[0, 1])), 3, rgba, thickness=-1)

        return frame

    def draw_corner_info(self, frame, x1, y1, info_str, line_color):
        (test_width, text_height), baseline = cv2.getTextSize(
            info_str, self.FONT,
            self.FONT_SCALE * 0.5,
            self.FONT_THICKNESS)
        cv2.rectangle(
            frame, (x1, y1 - text_height),
            (x1 + test_width, y1 + baseline), line_color, cv2.FILLED)
        cv2.putText(
            frame, info_str, (x1, y1), self.FONT,
            self.FONT_SCALE * 0.5, (0, 0, 0), self.FONT_THICKNESS,
            cv2.LINE_AA)
        return frame

    def draw_bev_canvas(self):
        # Set x, y limit and mark border
        self.ax.set_aspect('equal', adjustable='datalim')
        self.ax.set_xlim(self.x_min - 1, self.x_max + 1)
        self.ax.set_ylim(self.y_min - 1, self.y_max + 1)
        self.ax.tick_params(axis='both', labelbottom=False, labelleft=False)
        self.ax.xaxis.set_minor_locator(MultipleLocator(self.interval))
        self.ax.yaxis.set_minor_locator(MultipleLocator(self.interval))

        for radius in range(self.y_max, -1, -self.interval):
            # Mark all around sector
            self.ax.add_patch(
                mpatches.Wedge(
                    center=[0, 0],
                    alpha=0.1,
                    aa=True,
                    r=radius,
                    theta1=-180,
                    theta2=180,
                    fc="black"))

            # Mark range
            if radius / np.sqrt(2) + 8 < self.x_max:
                self.ax.text(
                    radius / np.sqrt(2) + 3,
                    radius / np.sqrt(2) - 5,
                    f'{radius}m',
                    rotation=-45,
                    color='darkblue',
                    fontsize='xx-large')

        # Mark visible sector
        self.ax.add_patch(
            mpatches.Wedge(
                center=[0, 0],
                alpha=0.1,
                aa=True,
                r=self.y_max,
                theta1=45,
                theta2=135,
                fc="cyan"))

        # Mark ego-vehicle
        self.ax.arrow(0, 0, 0, 3, color='black', width=0.5, overhang=0.3)

    def draw_2d_bbox(
            self,
            frame,
            box,
            line_color: tuple = (0, 255, 0),
            line_width: int = 3,
            corner_info: str = None):
        cv2.rectangle(
            frame, (box[0], box[1]), (box[2], box[3]), line_color,
            line_width)
        if corner_info is not None:
            x1 = int(box[0])
            y1 = int(box[3])
            frame = self.draw_corner_info(
                frame, x1, y1, corner_info, line_color)

        return frame

    def draw_3d_bbox(
            self,
            frame,
            points_camera,
            cam_calib,
            cam_pose,
            cam_near_clip: float = 0.15,
            line_color: tuple = (0, 255, 0),
            line_width: int = 3,
            corner_info: str = None):
        projpoints = tu.get_3d_bbox_vertex(
            cam_calib, cam_pose, points_camera, cam_near_clip)

        for p1, p2 in projpoints:
            cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
                     line_color, line_width)

        if corner_info is not None:
            is_before = False
            cp1 = tu.cameratoimage(points_camera[0:1], cam_calib)[0]

            if cp1 is not None:
                is_before = tu.is_before_clip_plane_camera(
                    points_camera[0:1], cam_near_clip)[0]

            if is_before:
                x1 = int(cp1[0])
                y1 = int(cp1[1])
                frame = self.draw_corner_info(
                    frame, x1, y1, corner_info, line_color)

        return frame

    def plot_3D_box(
            self, n_seq_pd: str, pd_seq: dict):
        id_to_color = {}
        cmap = RandomColor(len(pd_seq['frames']))

        # Variables
        subfolder = 'shows_3D' if self.draw_3d else 'shows_2D'
        self.trk_vid_name = os.path.join(
            self.res_folder, subfolder, f'{n_seq_pd}_tracking.mp4')
        self.bev_vid_name = os.path.join(
            self.res_folder, 'shows_BEV', f'{n_seq_pd}_birdsview.mp4')

        print(f"Trk: {self.trk_vid_name}")
        print(f"BEV: {self.bev_vid_name}")

        resH = pd_seq['height']
        resW = pd_seq['width']

        # set output video
        os.makedirs(os.path.dirname(self.trk_vid_name), exist_ok=True)
        os.makedirs(os.path.dirname(self.bev_vid_name), exist_ok=True)
        vid_trk = cv2.VideoWriter(
            self.trk_vid_name, self.FOURCC, self.fps,
            (resW, resH))
        vid_bev = cv2.VideoWriter(
            self.bev_vid_name, self.FOURCC, self.fps,
            (self.bev_size, self.bev_size))

        max_frames = len(pd_seq['frames'])
        print(f"Video ID: {n_seq_pd}\n"
              f"PD frames: {max_frames}")

        loc_world_hist_pd = {}
        for n_frame in range(max_frames):
            self.FOCAL_LENGTH = pd_seq['frames'][n_frame]['cam_calib'][0][0]

            pd_objects = pd_seq['frames'].get(n_frame, {'annotations': []})
            pd_annos = {}

            if n_frame % 100 == 0:
                print(f"Frame {n_frame} ...")

            # Get objects
            img = cv2.imread(pd_objects['im_path'])
            (test_width, text_height), baseline = cv2.getTextSize(
                str(n_frame), self.FONT, self.FONT_SCALE,
                self.FONT_THICKNESS * 2)
            cv2.rectangle(
                img, (0, 0),
                (test_width, text_height + baseline),
                (255, 255, 255), -1)
            cv2.putText(
                img, str(n_frame),
                (0, text_height + baseline // 2), self.FONT,
                self.FONT_SCALE, (0, 0, 0),
                self.FONT_THICKNESS * 2, cv2.LINE_AA)

            cam_coords = np.array(pd_objects['cam_loc'])
            cam_rotation = np.array(pd_objects['cam_rot'])
            cam_calib = np.array(pd_objects['cam_calib'])
            cam_pose = Pose(cam_coords, cam_rotation)

            if len(pd_objects['annotations']) > 0:
                pd_annos = sorted(
                    pd_objects['annotations'],
                    key=lambda x: x['location'][2],
                    reverse=True)

            for hypo in pd_annos:
                # Get information of gt and pd
                tid_pd_str = f"{hypo['track_id']}PD"
                tid_pd = hypo['track_id']
                box_pd = np.array(hypo['box']).astype(int)
                _, w_pd, l_pd = hypo['dimension']
                hypo_dict = self.get_3d_info(hypo, cam_calib, cam_pose)
                center_pd = hypo_dict['center']
                loc_world_pd = hypo_dict['loc_world']
                yaw_pd = hypo_dict['yaw']
                yaw_world_pd = hypo_dict['yaw_world_quat']
                box3d_pd = hypo_dict['box3d']
                obj_class_pd = hypo_dict['class']
                if tid_pd not in loc_world_hist_pd:
                    loc_world_hist_pd[tid_pd] = {
                        'loc': [loc_world_pd],
                        'yaw': [yaw_world_pd]
                    }
                elif len(loc_world_hist_pd[tid_pd]['loc']) > self.num_hist:
                    loc_world_hist_pd[tid_pd]['loc'] = \
                        loc_world_hist_pd[tid_pd]['loc'][1:] + [loc_world_pd]
                    loc_world_hist_pd[tid_pd]['yaw'] = \
                        loc_world_hist_pd[tid_pd]['yaw'][1:] + [yaw_world_pd]
                else:
                    loc_world_hist_pd[tid_pd]['loc'].append(loc_world_pd)
                    loc_world_hist_pd[tid_pd]['yaw'].append(yaw_world_pd)

                # Get box color
                # color is in BGR format (for cv2), color[:-1] in RGB format
                # (for plt)
                if tid_pd_str not in list(id_to_color):
                    id_to_color[tid_pd_str] = cmap.get_random_color(scale=255)
                color = id_to_color[tid_pd_str]

                if self.draw_tid:
                    info_str = f"{obj_class_pd}{tid_pd}PD"
                else:
                    info_str = f"{obj_class_pd}"

                # Make rectangle
                if self.draw_3d:
                    # Make rectangle
                    img = self.draw_3d_bbox(
                        img,
                        box3d_pd,
                        cam_calib,
                        cam_pose,
                        line_color=color,
                        corner_info=info_str)

                if self.draw_2d:
                    self.draw_2d_bbox(
                        img,
                        box_pd,
                        line_color=color,
                        line_width=3,
                        corner_info=info_str)

                if self.draw_traj:
                    # Draw trajectories
                    img = self.draw_3d_traj(
                        img,
                        loc_world_hist_pd[tid_pd]['loc'],
                        cam_calib,
                        cam_pose,
                        line_color=color)

                if self.draw_bev:
                    # Change BGR to RGB
                    color_bev = [c / 255.0 for c in color[::-1]]
                    center_hist_pd = tu.worldtocamera(
                        np.vstack(loc_world_hist_pd[tid_pd]['loc']),
                        cam_pose.position, cam_pose.rotation)[:, [0, 2]]
                    quat_cam_rot_t = Quaternion(matrix=cam_pose.rotation.T)
                    yaw_hist_pd = []
                    for quat_yaw_world_pd in loc_world_hist_pd[tid_pd]['yaw']:
                        rotation_cam = quat_cam_rot_t * quat_yaw_world_pd
                        vtrans = np.dot(
                            rotation_cam.rotation_matrix,
                            np.array([1, 0, 0]))
                        yaw_hist_pd.append(
                            -np.arctan2(vtrans[2], vtrans[0]).tolist())
                    yaw_hist_pd = np.vstack(yaw_hist_pd)
                    plot_bev_obj(
                        self.ax,
                        center_pd,
                        center_hist_pd,
                        yaw_pd,
                        yaw_hist_pd,
                        l_pd,
                        w_pd,
                        color_bev,
                        'PD',
                        line_width=2)

            # Plot
            if vid_trk and (self.draw_3d or self.draw_2d):
                vid_trk.write(cv2.resize(img, (resW, resH)))

            # Plot
            if self.draw_bev:
                self.draw_bev_canvas()
                if vid_bev:
                    vid_bev.write(fig2data(self.fig))
                    plt.cla()
                else:
                    self.fig.show()
                    plt.cla()

        vid_trk.release()
        vid_bev.release()

    def save_vid(self, info_pd):
        # Loop over save_range and plot the BEV
        logger.info("Total {} frames. Now saving...".format(
            sum([len(seq['frames']) for _, seq in info_pd.items()])))

        # Iterate through all objects
        for (n_seq_pd, pd_seq) in info_pd.items():
            # Plot annotation with predictions
            self.plot_3D_box(n_seq_pd, pd_seq)

            # Merge two video vertically
            if self.is_merge:
                subfolder = '/shows_3D/' if self.draw_3d else '/shows_2D/'
                output_name = self.trk_vid_name.replace(
                    subfolder, '/shows_compose/').replace('_tracking', '_compose')
                merge_vid(self.trk_vid_name, self.bev_vid_name, output_name)

        logger.info("Save done!")
