import os
from typing import List, Optional
from math import ceil

import onnxruntime
import cv2
import PIL
import PIL.Image
import numpy as np
from itertools import product as product
import torchvision
import torch
from tqdm import tqdm
from scipy.io import loadmat, savemat

# TODO: manage pathes with config class
PATH_TO_IMAGE_TO_COEFF_ONNX   = "/home/t-ibayashi/Workspace/ax/ailia-models/image_animation/sad_talker/resources/imageToCoeff.onnx"
PATH_TO_FACE_DETECTOR_ONNX    = "/home/t-ibayashi/Workspace/ax/ailia-models/image_animation/sad_talker/resources/faceDetector.onnx"
PATH_TO_FACE_ALIGNER_ONNX     = "/home/t-ibayashi/Workspace/ax/ailia-models/image_animation/sad_talker/resources/faceAligner.onnx"
PATH_TO_STANDARD_LANDMARK_MAT = "/home/t-ibayashi/Workspace/ax/ailia-models/image_animation/sad_talker/resources/similarity_Lm3D_all.mat"

class ImageProcessor:
    def __init__(
        self,
    ):
        self.image_to_coeff = ImageToCoeff()
        self.keypoint_extractor = KeypointExtractor()

    def generate(
        self,
        pic_path: str,
        preprocess: str,
        save_dir: str,
        input_size: int = 256,
    ):
        """
        generate coefficients for face model from image
        """
        pic_name = os.path.splitext(os.path.split(pic_path)[-1])[0]
        cropped_path   = os.path.join(save_dir, pic_name+'.png')
        coeff_path     = os.path.join(save_dir, pic_name+'.mat') 
        landmarks_path = os.path.join(save_dir, pic_name+'_landmarks.txt')

        if not os.path.isfile(pic_path):
            raise ValueError('pic_path must be a valid path to video/image file')

        elif pic_path.split('.')[-1] in ['jpg', 'png', 'jpeg']:
            # loader for first frame
            full_frames = [cv2.imread(pic_path)]
        else:
            raise Exception("Not supported yet")

        # convert to RGB
        x_full_frames= [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  for frame in full_frames]

        # 1. execute preprocess (resize or crop)
        if preprocess == "resize":
            # only resize input image
            oy1, oy2, ox1, ox2 = 0, x_full_frames[0].shape[0], 0, x_full_frames[0].shape[1] 
            crop_info = ((ox2 - ox1, oy2 - oy1), None, None)
        elif preprocess == "crop":
            # crop face
            raise Exception("Not supported yet")
        else:
            raise Exception("Invalid preprocess")

        frames_pil = [PIL.Image.fromarray(cv2.resize(frame,(input_size, input_size))) for frame in x_full_frames]
        if len(frames_pil) == 0:
            print('No face is detected in the input file')
            return None, None

        # save crop info
        for frame in frames_pil:
            # save first frame
            cv2.imwrite(cropped_path, cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
            break

        # 2. get the landmark according to the detected face
        if os.path.isfile(landmarks_path):
            print(' Using saved landmarks.')
            lmandmarks = np.loadtxt(landmarks_path).astype(np.float32)
            lmandmarks = lmandmarks.reshape([len(x_full_frames), -1, 2])
        else:
            lmandmarks = self.keypoint_extractor.extract_keypoints(frames_pil, landmarks_path)

        # 3. generate coeffitients for face model
        if os.path.isfile(coeff_path):
            # SKIP
            pass
        else:
            full_coeffs = []
            video_coeffs = []
            for idx in tqdm(range(len(frames_pil)), desc='3DMM Extraction In Video:'):
                frame = frames_pil[idx]
                lm = lmandmarks[idx].reshape([-1, 2])
                coeff, trans_params = self.image_to_coeff.get_coefficients(
                    image=frame,
                    landmarks=lm,
                )
                coeff_splited = self.image_to_coeff.split_coeff(
                    coeffs=coeff,
                )
                coeff_splited = np.concatenate(
                    [
                        coeff_splited['exp'], 
                        coeff_splited['angle'],
                        coeff_splited['trans'],
                        trans_params[2:][None],
                    ],
                    1,
                )
                video_coeffs.append(coeff_splited)
                full_coeffs.append(coeff)

            semantic = np.array(video_coeffs)[:,0]
            savemat(coeff_path, {'coeff_3dmm': semantic, 'full_3dmm': np.array(full_coeffs)[0]})
        return coeff_path, cropped_path, crop_info


class ImageToCoeff:
    def __init__(self) -> None:
        self.model = onnxruntime.InferenceSession(PATH_TO_IMAGE_TO_COEFF_ONNX, providers=["CPUExecutionProvider"])
        self.lm3d_std = self._load_lm3d(PATH_TO_STANDARD_LANDMARK_MAT)

    def _load_lm3d(
        self,
        landmark_path: str,
    ):
        """
        load landmarks for standard face, which is used for image preprocessing
        """
        Lm3D = loadmat(landmark_path)
        Lm3D = Lm3D['lm']

        # calculate 5 facial landmarks using 68 landmarks
        lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
        Lm3D = np.stack(
            [
                Lm3D[lm_idx[0], :], 
                np.mean(Lm3D[lm_idx[[1, 2]], :], 0),
                np.mean(Lm3D[lm_idx[[3, 4]], :], 0), Lm3D[lm_idx[5], :], Lm3D[lm_idx[6], :],
            ], axis=0)
        Lm3D = Lm3D[[1, 2, 0, 3, 4], :]
        return Lm3D

    def _extract_5p(self, lm):
        """
        utils for face reconstruction
        """
        lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
        lm5p = np.stack([lm[lm_idx[0], :], np.mean(lm[lm_idx[[1, 2]], :], 0), np.mean(
            lm[lm_idx[[3, 4]], :], 0), lm[lm_idx[5], :], lm[lm_idx[6], :]], axis=0)
        lm5p = lm5p[[1, 2, 0, 3, 4], :]
        return lm5p

    def _pos(self, xp, x):
        """
        calculating least square problem for image alignment
        """
        npts = xp.shape[1]

        A = np.zeros([2*npts, 8])

        A[0:2*npts-1:2, 0:3] = x.transpose()
        A[0:2*npts-1:2, 3] = 1

        A[1:2*npts:2, 4:7] = x.transpose()
        A[1:2*npts:2, 7] = 1

        b = np.reshape(xp.transpose(), [2*npts, 1])

        k, _, _, _ = np.linalg.lstsq(A, b)

        R1 = k[0:3]
        R2 = k[4:7]
        sTx = k[3]
        sTy = k[7]
        s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2
        t = np.stack([sTx, sTy], axis=0)

        return t, s

    def _resize_n_crop_img(self, img, lm, t, s, target_size=224., mask=None):
        """
        resize and crop images for face reconstruction
        """
        w0, h0 = img.size
        w = (w0*s).astype(np.int32)
        h = (h0*s).astype(np.int32)
        left = (w/2 - target_size/2 + float((t[0] - w0/2)*s)).astype(np.int32)
        right = left + target_size
        up = (h/2 - target_size/2 + float((h0/2 - t[1])*s)).astype(np.int32)
        below = up + target_size

        img = img.resize((w, h), resample=PIL.Image.BICUBIC)
        img = img.crop((left, up, right, below))

        if mask is not None:
            mask = mask.resize((w, h), resample=PIL.Image.BICUBIC)
            mask = mask.crop((left, up, right, below))

        lm = np.stack([lm[:, 0] - t[0] + w0/2, lm[:, 1] -t[1] + h0/2], axis=1)*s
        lm = lm - np.reshape(np.array([(w/2 - target_size/2), (h/2-target_size/2)]), [1, 2])
        return img, lm, mask

    def _align_img(
        self,
        img: PIL.Image.Image,
        lm: np.ndarray,
        lm3D: np.ndarray,
        mask: Optional[PIL.Image.Image]=None,
        target_size: int=224.,
        rescale_factor: int=102.,
    ):
        """
        Return:
            transparams        --numpy.array  (raw_W, raw_H, scale, tx, ty)
            img_new            --PIL.Image  (target_size, target_size, 3)
            lm_new             --numpy.array  (68, 2), y direction is opposite to v direction
            mask_new           --PIL.Image  (target_size, target_size)
        
        Parameters:
            img                --PIL.Image  (raw_H, raw_W, 3)
            lm                 --numpy.array  (68, 2), y direction is opposite to v direction
            lm3D               --numpy.array  (5, 3)
            mask               --PIL.Image  (raw_H, raw_W, 3)
        """

        w0, h0 = img.size
        if lm.shape[0] != 5:
            lm5p = self._extract_5p(lm)
        else:
            lm5p = lm

        # calculate translation and scale factors using 5 facial landmarks and standard landmarks of a 3D face
        t, s = self._pos(lm5p.transpose(), lm3D.transpose())
        s = rescale_factor/s

        # processing the image
        img_new, lm_new, mask_new = self._resize_n_crop_img(img, lm, t, s, target_size=target_size, mask=mask)
        trans_params = np.array([w0, h0, s, t[0], t[1]], dtype=object)
        return trans_params, img_new, lm_new, mask_new

    def split_coeff(
        self,
        coeffs: np.ndarray,
    ):
        id_coeffs = coeffs[:, :80]
        exp_coeffs = coeffs[:, 80: 144]
        tex_coeffs = coeffs[:, 144: 224]
        angles = coeffs[:, 224: 227]
        gammas = coeffs[:, 227: 254]
        translations = coeffs[:, 254:]
        return {
            'id': id_coeffs,
            'exp': exp_coeffs,
            'tex': tex_coeffs,
            'angle': angles,
            'gamma': gammas,
            'trans': translations
        }

    def foward(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        coefficients = self.model.run(
            None,
            {
                "image": image,
            }
        )[0]
        return coefficients

    def get_coefficients(
        self,
        image: PIL.Image.Image,
        landmarks: np.ndarray,
    ):
        W,H = image.size
        if np.mean(landmarks) == -1:
            landmarks = (self.lm3d_std[:, :2]+1)/2.
            landmarks = np.concatenate(
                [landmarks[:, :1]*W, landmarks[:, 1:2]*H], 1
            )
        else:
            landmarks[:, -1] = H - 1 - landmarks[:, -1]
        trans_params, im1, lm1, _ = self._align_img(
            img=image,
            lm=landmarks,
            lm3D=self.lm3d_std,
        )
        trans_params = np.array([float(item) for item in np.hsplit(trans_params, 5)]).astype(np.float32)
        im_t = np.array(im1) / 255.
        im_t = im_t.astype(np.float32)
        im_t = np.transpose(im_t, (2, 0, 1))
        im_t = np.expand_dims(im_t, axis=0)

        coefficients = self.foward(im_t)
        return coefficients, trans_params


class KeypointExtractor:
    def __init__(self):
        self.face_aligner = FaceAligner()
        self.face_detector = FaceDetector()

    def _landmark_98_to_68(
        self,
        landmark_98: np.ndarray,
    ):
        """Transfer 98 landmark positions to 68 landmark positions.
        Args:
            landmark_98(numpy array): Polar coordinates of 98 landmarks, (98, 2)
        Returns:
            landmark_68(numpy array): Polar coordinates of 98 landmarks, (68, 2)
        """

        landmark_68 = np.zeros((68, 2), dtype='float32')
        # cheek
        for i in range(0, 33):
            if i % 2 == 0:
                landmark_68[int(i / 2), :] = landmark_98[i, :]
        # nose
        for i in range(51, 60):
            landmark_68[i - 24, :] = landmark_98[i, :]
        # mouth
        for i in range(76, 96):
            landmark_68[i - 28, :] = landmark_98[i, :]
        # left eyebrow
        landmark_68[17, :] = landmark_98[33, :]
        landmark_68[18, :] = (landmark_98[34, :] + landmark_98[41, :]) / 2
        landmark_68[19, :] = (landmark_98[35, :] + landmark_98[40, :]) / 2
        landmark_68[20, :] = (landmark_98[36, :] + landmark_98[39, :]) / 2
        landmark_68[21, :] = (landmark_98[37, :] + landmark_98[38, :]) / 2
        # right eyebrow
        landmark_68[22, :] = (landmark_98[42, :] + landmark_98[50, :]) / 2
        landmark_68[23, :] = (landmark_98[43, :] + landmark_98[49, :]) / 2
        landmark_68[24, :] = (landmark_98[44, :] + landmark_98[48, :]) / 2
        landmark_68[25, :] = (landmark_98[45, :] + landmark_98[47, :]) / 2
        landmark_68[26, :] = landmark_98[46, :]
        # left eye
        LUT_landmark_68_left_eye = [36, 37, 38, 39, 40, 41]
        LUT_landmark_98_left_eye = [60, 61, 63, 64, 65, 67]
        for idx, landmark_98_index in enumerate(LUT_landmark_98_left_eye):
            landmark_68[LUT_landmark_68_left_eye[idx], :] = landmark_98[landmark_98_index, :]
        # right eye
        LUT_landmark_68_right_eye = [42, 43, 44, 45, 46, 47]
        LUT_landmark_98_right_eye = [68, 69, 71, 72, 73, 75]
        for idx, landmark_98_index in enumerate(LUT_landmark_98_right_eye):
            landmark_68[LUT_landmark_68_right_eye[idx], :] = landmark_98[landmark_98_index, :]

        return landmark_68

    def extract_keypoint(
        self,
        image: PIL.Image.Image,
        landmarks_path: Optional[str] = None,    
    ) -> np.ndarray:
        """
        extrace keypoints from single image

        Args:
            image (PIL.Image.Image): image to extract keypoints
            landmarks_path (Optional[str]): path to save landmarks. Defaults to None.

        Return:
            keypoints (np.ndarray): extracted keypoints from image
        """
        if isinstance(image, list):
            raise Exception("image should be a single Image instance. You should use extract_keypoints.")

        with torch.no_grad():
            bboxes = self.face_detector.detect_face(image, 0.97)
            bboxes = bboxes[0]
            image_np = np.array(image)
            image_np = image_np[int(bboxes[1]):int(bboxes[3]), int(bboxes[0]):int(bboxes[2]), :]

            landmarks = self.face_aligner.get_landmarks(image_np)
            keypoints = self._landmark_98_to_68(landmarks)

            #### keypoints to the original location
            keypoints[:,0] += int(bboxes[0])
            keypoints[:,1] += int(bboxes[1])

        if landmarks_path is not None:
            np.savetxt(os.path.splitext(landmarks_path)[0]+'.txt', keypoints.reshape(-1))
        return keypoints

    def extract_keypoints(
        self,
        images: List[PIL.Image.Image],
        landmarks_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        extrace keypoints from multiple images

        Args:
            images (List[PIL.Image.Image]): images to extract keypoints
            landmarks_path (Optional[str]): path to save landmarks. Defaults to None.

        Return:
            keypoints (np.ndarray): extracted keypoints from image
        """
        keypoints = []
        for image in tqdm(images,desc='landmark Det:'):
            current_kp = self.extract_keypoint(image)
            if np.mean(current_kp) == -1 and keypoints:
                keypoints.append(keypoints[-1])
            else:
                keypoints.append(current_kp[None])
        keypoints = np.concatenate(keypoints, 0)
        np.savetxt(os.path.splitext(landmarks_path)[0]+'.txt', keypoints.reshape(-1))
        return keypoints


class FaceDetector:
    def __init__(self):
        self.model = onnxruntime.InferenceSession(PATH_TO_FACE_DETECTOR_ONNX, providers=["CPUExecutionProvider"])

    def foward(
        self,
        image: np.ndarray,
    ):

        # reshape
        input_img = image.transpose(2, 0, 1)      # (INPUT_SIZE,INPUT_SIZE,3)  → (3, INPUT_SIZE,INPUT_SIZE)
        input_img = np.expand_dims(input_img, axis=0) # (3, INPUT_SIZE,INPUT_SIZE) → (1,3,INPUT_SIZE,INPUT_SIZE)

        # RGB correction
        mean_tensor = np.array([[[[104.]], [[117.]], [[123.]]]], dtype=np.float32)
        input_img = input_img - mean_tensor

        # detect face
        location, confidence, landmarks = self.model.run(
            None,
            {
                "image": input_img,
            }
        )
        # generae default anchor box
        priorbox = PriorBox(image_size=input_img.shape[2:])
        priors = priorbox.forward()
        return location, confidence, landmarks, priors

    def _decode_bboxes(self, loc, priors, variances):
        """
        Decode the predicted bounding box locations based on the offset regression encoding performed during training.
        Args:
            loc (ndarray): Predicted location results from the loc layers, shape: [num_priors, 4]
            priors (ndarray): Prior boxes in center-offset form, shape: [num_priors, 4]
            variances: (list[float]) Variances of the prior boxes
        Return:
            Decoded bounding box predictions
        """
        boxes = np.concatenate(
            (   
                priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                priors[:, 2:] * np.exp(loc[:, 2:] * variances[1]),
            ),
            1,
        )
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def _decode_landm(self, pre, priors, variances):
        """Decode landm from predictions using priors to undo
        the encoding we did for offset regression at train time.
        Args:
            pre (tensor): landm predictions for loc layers,
                Shape: [num_priors,10]
            priors (tensor): Prior boxes in center-offset form.
                Shape: [num_priors,4].
            variances: (list[float]) Variances of priorboxes
        Return:
            decoded landm predictions
        """
        tmp = (
            priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
        )
        landms = np.concatenate(tmp, axis=1)
        return landms

    def _py_cpu_nms(self, dets, thresh):
        """Pure Python NMS baseline."""
        keep = torchvision.ops.nms(
            boxes=torch.Tensor(dets[:, :4]),
            scores=torch.Tensor(dets[:, 4]),
            iou_threshold=thresh,
        )
        return list(keep.numpy())

    def detect_face(
        self,
        image: PIL.Image.Image,
        conf_threshold: float=0.8,
        nms_threshold: float=0.4,
    ):
        # params
        variances=[0.1, 0.2]

        # convert to numpy array (BGR)
        image_np = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        image_np = image_np.astype(np.float32)
    
        # detect face position
        height, width = image_np.shape[0], image_np.shape[1]
        scale_box = np.array([width, height, width, height], dtype=np.float32)
        scale_landmark = np.array([width, height, width, height, width, height, width, height, width, height], dtype=np.float32)

        location, confidence, landmarks, priors = self.foward(image_np)
        boxes = self._decode_bboxes(np.squeeze(location), priors, variances=variances)
        boxes = boxes * scale_box

        scores = np.squeeze(confidence, axis=0)[:, 1]
        landmarks = self._decode_landm(landmarks.squeeze(0), priors, variances)
        landmarks = landmarks * scale_landmark

        # ignore low scores
        inds = np.where(scores > conf_threshold)[0]
        boxes, landmarks, scores = boxes[inds], landmarks[inds], scores[inds]

        # sort
        order = scores.argsort()[::-1]
        boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]

        # do NMS
        bounding_boxes = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = self._py_cpu_nms(bounding_boxes, nms_threshold)
        bounding_boxes, landmarks = bounding_boxes[keep, :], landmarks[keep]
        return np.concatenate((bounding_boxes, landmarks), axis=1)


class PriorBox(object):
    def __init__(self, image_size=None):
        super(PriorBox, self).__init__()
        self.min_sizes = [[16, 32], [64, 128], [256, 512]]
        self.steps = [8, 16, 32]
        self.clip = False
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)] for step in self.steps]
        self.name = 's'

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = np.reshape(anchors, (-1, 4))
        if self.clip:
            np.clip(output, a_min=0, a_max=1, out=output)
        return output


class FaceAligner:
    def __init__(self):
        self.model = onnxruntime.InferenceSession(PATH_TO_FACE_ALIGNER_ONNX, providers=["CPUExecutionProvider"])
        pass

    def _calculate_points(
        self,
        heatmaps: np.ndarray,
    ):
        # change heatmaps to landmarks
        B, N, H, W = heatmaps.shape
        HW = H * W
        BN_range = np.arange(B * N)

        heatline = heatmaps.reshape(B, N, HW)
        indexes = np.argmax(heatline, axis=2)

        preds = np.stack((indexes % W, indexes // W), axis=2)
        preds = preds.astype(float, copy=False)

        inr = indexes.ravel()

        heatline = heatline.reshape(B * N, HW)
        x_up = heatline[BN_range, inr + 1]
        x_down = heatline[BN_range, inr - 1]
        # y_up = heatline[BN_range, inr + W]

        if any((inr + W) >= 4096):
            y_up = heatline[BN_range, 4095]
        else:
            y_up = heatline[BN_range, inr + W]
        if any((inr - W) <= 0):
            y_down = heatline[BN_range, 0]
        else:
            y_down = heatline[BN_range, inr - W]

        think_diff = np.sign(np.stack((x_up - x_down, y_up - y_down), axis=1))
        think_diff *= .25

        preds += think_diff.reshape(B, N, 2)
        preds += .5
        return preds

    def foward(
        self,
        x: np.ndarray,

    ):
        x = x.astype(np.float32)
        result = self.model.run(
            None,
            {
                "image": x,
            }
        )
        outputs = result[0:4]
        boundary_channels = result[4:]
        return outputs, boundary_channels

    def get_landmarks(
        self,
        image: np.ndarray,
    ):
        H, W, _ = image.shape
        offset = W / 64, H / 64, 0, 0
        image = cv2.resize(image, (256, 256))
        inp = image[..., ::-1]
        inp = np.ascontiguousarray(inp.transpose((2, 0, 1)))

        inp = inp / 255.0
        inp = np.expand_dims(inp, axis=0)

        outputs, _ = self.foward(inp)

        heatmaps = outputs[-1][:, :-1, :, :]
        pred = self._calculate_points(heatmaps).reshape(-1, 2)
        pred *= offset[:2]
        pred += offset[-2:]
        return pred
