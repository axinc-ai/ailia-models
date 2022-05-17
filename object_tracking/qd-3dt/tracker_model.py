import numpy as np

onnx = False


class LSTM3DTracker(object):
    """
    This class represents the internel state of individual tracked objects
    observed as bbox.
    """
    count = 0

    def __init__(self, lstm_pred, lstm_refine, loc_dim, bbox3D, info):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        # coord3d - array of detections [x,y,z,theta,l,w,h]
        # X,Y,Z,theta, l, w, h, dX, dY, dZ

        self.loc_dim = loc_dim
        self.id = LSTM3DTracker.count
        LSTM3DTracker.count += 1

        self.nfr = 5
        self.hits = 1
        self.hit_streak = 0
        self.time_since_update = 0
        self.init_flag = True
        self.age = 0

        self.obj_state = np.hstack([bbox3D.reshape((7,)), np.zeros((3,))])
        self.history = np.tile(
            np.zeros_like(bbox3D[:self.loc_dim]), (self.nfr, 1))
        self.ref_history = np.tile(bbox3D[:self.loc_dim], (self.nfr + 1, 1))
        self.avg_angle = bbox3D[3]
        self.avg_dim = np.array(bbox3D[4:])
        self.prev_obs = bbox3D.copy()
        self.prev_ref = bbox3D[:self.loc_dim].copy()
        self.info = info

        self.lstm_pred = lstm_pred
        self.lstm_refine = lstm_refine
        self.batch_size = 1
        self.feature_dim = 64
        self.hidden_size = 128
        self.num_layers = 2

        self.hidden_pred = self.init_hidden()
        self.hidden_ref = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (
            np.zeros((self.num_layers, self.batch_size, self.hidden_size), dtype=np.float32),
            np.zeros((self.num_layers, self.batch_size, self.hidden_size), dtype=np.float32)
        )

    @staticmethod
    def fix_alpha(angle: float) -> float:
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def update_array(
            origin_array: np.ndarray,
            input_array: np.ndarray) -> np.ndarray:
        new_array = origin_array.copy()
        new_array[:-1] = origin_array[1:]
        new_array[-1:] = input_array
        return new_array

    def _init_history(self, bbox3D):
        self.ref_history = self.update_array(self.ref_history, bbox3D)
        self.history = np.tile([
            self.ref_history[-1] - self.ref_history[-2]],
            (self.nfr, 1))
        self.prev_ref[:self.loc_dim] = self.obj_state[:self.loc_dim]
        if self.loc_dim > 3:
            self.avg_angle = self.fix_alpha(
                self.ref_history[:, 3]).mean(axis=0)
            self.avg_dim = self.ref_history.mean(axis=0)[4:]
        else:
            self.avg_angle = self.prev_obs[3]
            self.avg_dim = np.array(self.prev_obs[4:])

    def _update_history(self, bbox3D):
        self.ref_history = self.update_array(self.ref_history, bbox3D)
        self.history = self.update_array(
            self.history, self.ref_history[-1] - self.ref_history[-2])
        # align orientation history
        self.history[:, 3] = self.history[-1, 3]
        self.prev_ref[:self.loc_dim] = self.obj_state[:self.loc_dim]
        if self.loc_dim > 3:
            self.avg_angle = self.fix_alpha(
                self.ref_history[:, 3]).mean(axis=0)
            self.avg_dim = self.ref_history.mean(axis=0)[4:]
        else:
            self.avg_angle = self.prev_obs[3]
            self.avg_dim = np.array(self.prev_obs[4:])

    def update(self, bbox3D, info):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        if self.age == 1:
            self.obj_state[:self.loc_dim] = bbox3D[:self.loc_dim].copy()

        if self.loc_dim > 3:
            # orientation correction
            self.obj_state[3] = self.fix_alpha(self.obj_state[3])
            bbox3D[3] = self.fix_alpha(bbox3D[3])

            # if the angle of two theta is not acute angle
            # make the theta still in the range
            curr_yaw = bbox3D[3]
            if np.pi / 2.0 < abs(curr_yaw -
                                 self.obj_state[3]) < np.pi * 3 / 2.0:
                self.obj_state[3] += np.pi
                if self.obj_state[3] > np.pi:
                    self.obj_state[3] -= np.pi * 2
                if self.obj_state[3] < -np.pi:
                    self.obj_state[3] += np.pi * 2

            # now the angle is acute: < 90 or > 270,
            # convert the case of > 270 to < 90
            if abs(curr_yaw - self.obj_state[3]) >= np.pi * 3 / 2.0:
                if curr_yaw > 0:
                    self.obj_state[3] += np.pi * 2
                else:
                    self.obj_state[3] -= np.pi * 2

        location = self.obj_state[:self.loc_dim].reshape(1, self.loc_dim).astype(np.float32)
        observation = bbox3D[:self.loc_dim].reshape(1, self.loc_dim).astype(np.float32)
        prev_location = self.prev_ref[:self.loc_dim].reshape(1, self.loc_dim).astype(np.float32)
        confidence = info.reshape(1, 1).astype(np.float32)
        h_0, c_0 = self.hidden_ref
        if not onnx:
            output = self.lstm_refine.predict([
                location, observation, prev_location, confidence,
                h_0, c_0
            ])
        else:
            output = self.lstm_refine.run(
                None,
                {'location': location, 'observation': observation,
                 'prev_location': prev_location, 'confidence': confidence,
                 'h_0': h_0, 'c_0': c_0})
        refined_loc, h_1, c_1 = output
        self.hidden_ref = (h_1, c_1)

        refined_obj = refined_loc.flatten()
        if self.loc_dim > 3:
            refined_obj[3] = self.fix_alpha(refined_obj[3])

        self.obj_state[:self.loc_dim] = refined_obj
        self.prev_obs = bbox3D

        if np.pi / 2.0 < abs(bbox3D[3] - self.avg_angle) < np.pi * 3 / 2.0:
            for r_indx in range(len(self.ref_history)):
                self.ref_history[r_indx][3] = self.fix_alpha(
                    self.ref_history[r_indx][3] + np.pi)

        if self.init_flag:
            self._init_history(refined_obj)
            self.init_flag = False
        else:
            self._update_history(refined_obj)

        self.info = info

    def predict(self, update_state: bool = True):
        """
        Advances the state vector and returns the predicted bounding box
        estimate.
        """
        vel_history = self.history[..., :self.loc_dim].reshape(self.nfr, -1, self.loc_dim)
        vel_history = vel_history.astype(dtype=np.float32)
        location = self.obj_state[:self.loc_dim].reshape(-1, self.loc_dim)
        location = location.astype(dtype=np.float32)
        h_0, c_0 = self.hidden_pred
        if not onnx:
            output = self.lstm_pred.predict([
                vel_history, location,
                h_0, c_0
            ])
        else:
            output = self.lstm_pred.run(
                None, {'vel_history': vel_history, 'location': location, 'h_0': h_0, 'c_0': c_0})
        pred_loc, h_1, c_1 = output
        hidden_pred = (h_1, c_1)

        pred_state = self.obj_state.copy()
        pred_state[:self.loc_dim] = pred_loc.flatten()
        pred_state[7:] = pred_state[:3] - self.prev_ref[:3]
        if self.loc_dim > 3:
            pred_state[3] = self.fix_alpha(pred_state[3])

        if update_state:
            self.hidden_pred = hidden_pred
            self.obj_state = pred_state

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return pred_state.flatten()

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.obj_state.flatten()
