"""
post for crosswalk detection, the vector crossing method used to detect the vehicle crossing crosswalk behavior
"""
import copy

import cv2
import numpy as np


class DmPost(object):
    """damei post"""

    def __init__(self, clrl):
        self.crossout = self.init_crossout()
        self.vetors = self.init_vectors()
        self.old_vectors, self.frame_thresh, self.old_fileorder = \
            self.init_jitter_pretection_params()  # Used for jitter protection
        self.rect = cv2.imread("settings/rect.jpg")
        self.t_inter = self.init_others()
        self.control_line = clrl['control_line']
        self.red_line = clrl['red_line']
        self.det_id = 0
        self.v_idx = 0

    def init_crossout(self):
        field_size = 5

        crossout = np.zeros((field_size, 8))
        # 5 rows. 8 columns:
        #  out_index, detect_id, fileorder, is crosswalk exists, xc, yc, count，recording_flag
        crossout[:, :] = -2  # -2 represents undetected
        crossout[:, 0] = range(len(crossout))  # init out_index
        crossout[:, 6:8] = 0  # init count and recording_flag to 0

        return crossout

    def init_vectors(self):
        vector_size = 600 * 30  # maximun 600 seconds, sampling ratio 30。
        vectors = np.zeros((vector_size, 2))  # 2: store xc, yc
        vectors[:, :] = -2  # init

        return vectors

    def init_rect(self, shape):
        h, w = shape[:2]
        scale_factor = 3.5
        resized_width = int(w / scale_factor)
        rect_resized_size = (resized_width, 200)
        self.rect = cv2.resize(self.rect, rect_resized_size, interpolation=cv2.INTER_LINEAR)

    def init_jitter_pretection_params(self, time_thresh=2):
        # The sampling rate of the original video, 1 frame corresponds to 1/25 = 40ms
        raw_video_sampling_rate = 25

        # The sampling interval for converting the original video to the video. If it is 1, full sampling,
        # 1 frame still corresponds to 1/25, if it is 5, 1 frame corresponds to 1/(25/5) = 200ms
        video2imgs_sampling_interval = 5

        # frame thresh, which is 10.
        ft = int(time_thresh / (video2imgs_sampling_interval / raw_video_sampling_rate))

        return None, ft, -ft

    def init_others(self):
        FPS = 30  # 1s has 25 pictures
        sampling_ratio = 30  # 1 sample for every 4
        sampling_rate = FPS / sampling_ratio  # Sampling rate, 5 frames per second
        base_time = 0  # Base time, when the detected image exceeds the chunk size, the base time should be increased
        t_inter = 1 / sampling_rate  # time interpolation 0.2 seconds per photo

        return t_inter

    def imgAdd(self, small_img, big_image, x, y, alpha=0.5):
        row, col = small_img.shape[:2]
        if small_img.shape[0] > big_image.shape[0] or small_img.shape[1] > big_image.shape[1]:
            raise NameError(f'imgAdd, the size of small img bigger than big img.')
        roi = big_image[x:x + row, y:y + col, :]
        roi = cv2.addWeighted(small_img, alpha, roi, 1 - alpha, 0)
        big_image[x:x + row, y:y + col] = roi
        return big_image

    def imgputText(self, img, txt, pos, lt, tf):
        cv2.putText(img, txt, pos, 0, lt / 3, (30, 30, 224), thickness=tf, lineType=cv2.LINE_AA)

    def imgplotVectors(self, img, vt):
        if vt.shape[0] == 0:
            return
        for i in range(vt.shape[0] - 1):
            pt1, pt2 = tuple(vt[i]), tuple(vt[i + 1])
            cv2.arrowedLine(img, pt1, pt2, (0, 0, 255), 5, 8, 0, 0.3)

    def dmpost(self, img, det, names, cls='crosswalk'):
        crossout = self.crossout
        vectors = self.vetors
        det_id = self.det_id
        self.det_id += 1

        # Draw upper and lower control lines
        cl = self.control_line
        rl = self.red_line
        lt = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
        tf = max(lt - 1, 1)  # 2
        h, w = img.shape[:2]
        top = [(0, cl[0]), (w, cl[0])]
        button = [(0, cl[1]), (w, cl[1])]
        middle = [(int(w / 4), int(cl[0] + (cl[1] - cl[0]) * rl)), (int(3 * w / 4), int(cl[0] + (cl[1] - cl[0]) * rl))]
        arrow = [(int(w / 2), middle[0][1]), (int(w / 2), middle[0][1] + 50)]
        textpos = (top[0][0] + 10, top[0][1] + 30)

        cv2.line(img, top[0], top[1], color=(30, 30, 224), thickness=lt)
        cv2.line(img, button[0], button[1], color=(30, 30, 224), thickness=lt)
        cv2.line(img, middle[0], middle[1], color=(120, 85, 220), thickness=lt)  # pink
        cv2.arrowedLine(img, arrow[0], arrow[1], (120, 85, 220), 5, 8, 0, 0.3)
        cv2.putText(img, 'Control Line', textpos, 0, lt / 3, (30, 30, 224), thickness=tf, lineType=cv2.LINE_AA)

        cls_idx = names.index(cls)  # class index i.e. 0 for crosswalk
        det_include_class = cls_idx in det[:, 5] if det is not None else False
        if det is not None and det_include_class:
            crosswalk = np.zeros((det[det[:, 5] == 0].shape[0], 8))
            # [n, 6] becomes [n, 8], remove cls.
            # 8 are x1, y1, x2, y2, conf, cx, cy, is_in_control_line

            # As long as the selection is crosswalk, the default starting number is 0, 0 1 2 3 4
            crosswalk[:, :5] = det[det[:, 5] == 0][:, :5]
            cx = (crosswalk[:, 0] + crosswalk[:, 2]) // 2
            cy = (crosswalk[:, 1] + crosswalk[:, 3]) // 2
            crosswalk[:, 5] = cx
            crosswalk[:, 6] = cy
            is_in_cl = (crosswalk[:, 6] > cl[0]) | (crosswalk[:, 6] < cl[1])
            crosswalk[:, 7] = is_in_cl

            if crosswalk.shape[0] > 1:
                # Multiple zebra crossings are detected at the same time,
                # according to the previously recorded crossout,
                # determine which one to choose.
                last_co = crossout[crossout[:, 3] != -2]  # last crossout
                if len(last_co) == 0:  # if empty, then use the first one
                    valid_idx = 0
                else:
                    # Calculate the center of the detected zebra crossing
                    # which is close to the center of the last record.
                    lastcxy = last_co[-1][4:6]
                    currentcxy = crosswalk[:, 5:7]
                    distances = np.sum(np.square(currentcxy - lastcxy), axis=1)  # distance
                    valid_idx = np.argmin(distances)

                print(f'WANING: the detected crosswalk is more than one, use the {valid_idx + 1} one')
                crosswalk = crosswalk[valid_idx, :].reshape(1, -1)
                det = det[valid_idx, :].reshape(1, -1)
        else:
            crosswalk = np.zeros((1, 8))

        if det_id < crossout.shape[0]:  # detected_img id < 5
            # Update value of this column n columns:
            # out_index, detect_id, fileorder, with or without zebra crossing, xc, yc
            crossout[det_id, 1] = det_id
            crossout[det_id, 2] = det_id
            crossout[det_id, 3] = crosswalk[0, 7]
            crossout[det_id, 4:6] = crosswalk[0, 5: 7]  # xc, yc
            index = det_id
        else:
            # All rows except the serial number column are shifted up by one grid
            crossout[0:-1:, 1::] = crossout[1::, 1::]
            # Last column update value
            crossout[-1, 1] = det_id
            crossout[-1, 2] = det_id
            crossout[-1, 3] = crosswalk[0, 7]
            crossout[-1, 4:6] = crosswalk[0, 5: 7]  # xc, yc
            index = len(crossout) - 1

        exist, vector, scale = self.decode_crossout(crossout, index)
        recording = crossout[index, 7]

        if recording == 1 and vector is not None:
            vectors[self.v_idx, :] = vector[0]  # Write the first point of the vector
            vectors[self.v_idx + 1, :] = vector[1]  # Write the second point of the vector
            self.v_idx += 1
        elif recording == 1 and vector is None:
            # Record but no vector is passed in, keep the original
            pass
        else:
            if vectors[0, 0] != -2:
                # Save to another variable before initialization
                self.old_vectors = copy.deepcopy(vectors)
            vectors[:, :] = -2  # Initialize again
            self.v_idx = 0

        speed = None if scale is None else float((vector[1][1] - vector[0][1]) / (self.t_inter * scale))
        # The y distance of the vector is divided by the scale and then divided by the time interpolate

        # Draw results
        self.init_rect(img.shape)
        rect_pos = (20, 20)
        img = self.imgAdd(self.rect, img, rect_pos[1], rect_pos[0], alpha=0.5)

        pos = (20 + 20, 20 + 20 + 30)
        self.imgputText(img, f'crosswalk: {exist}', pos, lt, tf)
        pos = (20 + 20, 20 + 20 + 30 + 40)
        if speed is not None:
            self.imgputText(img, f'speed: {speed:.2f}', pos, lt, tf)
        else:
            self.imgputText(img, f'speed: {speed}', pos, lt, tf)

        vt = vectors[vectors[:, 0] != -2].astype(int)  # Filter t
        if vector is not None:
            self.imgplotVectors(img, vt)

        # Count, when vectors have values, the vector passes through the middle line,
        # and the current vector is not None, count+1
        crossout[index, 6] = crossout[index - 1, 6]  # Synchronize first, the count is the same as the previous one
        if vt.shape[0] != 0:
            # It is possible to cross the line if it is not empty,
            # and it is impossible to cross the line if it is empty.
            if vt[0, 1] < np.mean(cl):  # The starting point is above the control line
                intersect = vt[vt[:, 1] > middle[0][1]]  # All points whose y coordinate is greater than cl
                # There is a problem with only using the number of crossings equal to 1.
                # If the previous frame just crosses and the next frame jitters back,
                # it will be counted again. Add the condition inter2
                intersect2 = vt[-1, 1] > middle[0][1]  # Last intersection

                if intersect.shape[0] == 1 and vector is not None and intersect2:
                    # Add a jitter protection mechanism.
                    # When the frame counted this time is less than 10 frames compared with the previous one
                    # (corresponding to 2s), combine the current vectors with the previous vectors
                    c_fileorder = int(crossout[index, 2])
                    if (c_fileorder - self.old_fileorder) < self.frame_thresh:  # 10
                        print('\nxxx', c_fileorder, self.old_fileorder, self.frame_thresh)
                        vto = self.old_vectors[self.old_vectors[:, 0] != -2].astype(int)
                        new_vt = np.concatenate((vto, vt), axis=0)
                        self.imgplotVectors(img, new_vt)
                        vectors[:new_vt.shape[0], :] = new_vt
                    else:
                        crossout[index, 6] += 1  # count+1
                        prt_str = \
                            f'\n > The vehicle crossed a crosswalk!!' \
                            f' count+1, conf: {crosswalk[0, 4]:.2f},' \
                            f' current count: {int(crossout[index, 6])}.'
                        print(prt_str)
                        self.old_fileorder = copy.deepcopy(c_fileorder)

        count = int(crossout[index, 6])

        pos = (20 + 20, 20 + 20 + 30 + 40 + 40)
        self.imgputText(img, f'count: {count}', pos, lt, tf)

        # when there are vectors, it is crossing
        pos = (20 + 20, 20 + 20 + 30 + 40 + 40 + 40)
        status = 'No crosswalk' if vt.shape[0] == 0 else 'Crossing'
        self.imgputText(img, f'status: {status}', pos, lt, tf)

        prt_str = f' > detect_id: {det_id} speed: {speed} count: {count} status: {status}'
        print(prt_str)

        return img

    def decode_crossout(self, crossout, index, vector_threash=20, vector_max_threash=600):
        """
        Decode crossout, output whether there is a zebra crossing in the current image,
        zebra crossing displacement vector, time scale (spacing between indexes).
        Count counting algorithm: When the vector passes through the center line, count +1.
        """
        exist = crossout[index, 3]
        co = crossout[crossout[:, 3] == 1]  # All existing rows
        if exist == 0:
            if co.shape[0] == 0:  # There is no zebra crossing in the wild range
                crossout[:, 7] = 0  # recording 0
            return False, None, None
        else:
            if co.shape[0] == 1:  # Only the last line has
                crossout[:, 7] = 0  # recording 0
                return False, None, None
            else:
                scale = co[-1, 1] - co[-2, 1]  # detected_id difference
                vector = [co[-2, 4:6], co[-1, 4:6]]  # 2 points ((xc1, yc1), (xc2, yc2))
                vector2 = vector[1] - vector[0]  # (x2-x1) (y2-y1)
                length = np.sqrt(vector2[0] ** 2 + vector2[1] ** 2)
                y_shift = vector2[1]

                if length > vector_threash and y_shift < 0:
                    crossout[:, 7] = 0  # recording 0
                elif length > vector_max_threash:
                    # Sometimes there will be super long pixels greater than 300 pixels, filtered out,
                    # about ((680-400)*2/3)**2 680 and 400 are the control lines.
                    crossout[:, 7] = 0
                else:
                    crossout[:, 7] = 1  # recording 1

                return True, vector, scale
