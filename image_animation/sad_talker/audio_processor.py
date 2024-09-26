from typing import Optional
import os
import random

import librosa
import numpy as np
from scipy import signal
import scipy.io as scio
from scipy.signal import savgol_filter
from tqdm import tqdm
import onnxruntime

AUDIO2EXP_PATH = "/home/t-ibayashi/Workspace/ax/ailia-models/image_animation/sad_talker/resources/audio2Exp.onnx"
AUDIO2POSE_ENCODER_PATH = "/home/t-ibayashi/Workspace/ax/ailia-models/image_animation/sad_talker/resources/audio2Pose_audioEncoder.onnx"
AUDIO2POSE_NETG_PATH = "/home/t-ibayashi/Workspace/ax/ailia-models/image_animation/sad_talker/resources/audio2Pose_netG.onnx"
class AudioProcessorParams:
    def __init__(self):
        self.sample_rate = 16000
        self.num_mels = 80
        self.n_fft = 800
        self.hop_size = 200
        self.win_size = 800
        self.fmin = 55
        self.fmax = 7600
        self.ref_level_db = 20
        self.preemphasis = 0.97
        self.preemphasize = True
        self.frame_shift_ms = None
        self.min_level_db = -100
        self.signal_normalization = True
        self.allow_clipping_in_normalization = True
        self.symmetric_mels = True
        self.max_abs_value = 4.0

class AudioProcessor:
    def __init__(self):
        self.hp = AudioProcessorParams()
        self.mel_basis = None

        self.audio2exp_model = AudioToExp()
        self.audio_to_pose = AudioToPose()
        pass

    def _load_wav(self, path: str, sr: int)-> np.ndarray:
        """load wav file

        Args:
            path (str): Path to the wav file
            sr (int): Sampling rate

        Returns:
            _type_: _description_
        """        
        return librosa.core.load(path, sr=sr)[0]

    def _parse_audio_length(self, audio_length: str, sr: int, fps: int):
        bit_per_frames = sr / fps

        num_frames = int(audio_length / bit_per_frames)
        audio_length = int(num_frames * bit_per_frames)

        return audio_length, num_frames

    def _crop_pad_audio(self, wav: np.ndarray, audio_length: int):
        if len(wav) > audio_length:
            wav = wav[:audio_length]
        elif len(wav) < audio_length:
            wav = np.pad(wav, [0, audio_length - len(wav)], mode='constant', constant_values=0)
        return wav

    def _get_hop_size(self):
        hop_size = self.hp.hop_size
        if self.hp.hop_size is None:
            assert self.hp.frame_shift_ms is not None
            hop_size = int(self.hp.frame_shift_ms / 1000 * self.hp.sample_rate)
        return hop_size

    def _stft(self, y):
        return librosa.stft(y=y, n_fft=self.hp.n_fft, hop_length=self._get_hop_size(), win_length=self.hp.win_size)
 
    def _preemphasis(self, wav: np.ndarray, k: int, preemphasize: bool=True):
        if preemphasize:
            return signal.lfilter([1, -k], [1], wav)
        return wav

    def _build_mel_basis(self):
        assert self.hp.fmax <= self.hp.sample_rate // 2
        return librosa.filters.mel(
            sr=self.hp.sample_rate, n_fft=self.hp.n_fft, n_mels=self.hp.num_mels,
            fmin=self.hp.fmin, fmax=self.hp.fmax)

    def _amp_to_db(self, x):
        min_level = np.exp(self.hp.min_level_db / 20 * np.log(10))
        return 20 * np.log10(np.maximum(min_level, x))

    def _linear_to_mel(self, spectogram):
        # NOTE: AudioProcessor.mel_basis is not thread safe
        if self.mel_basis is None:
            self.mel_basis = self._build_mel_basis()
        return np.dot(self.mel_basis, spectogram)

    def _normalize(self, S):
        if self.hp.allow_clipping_in_normalization:
            if self.hp.symmetric_mels:
                return np.clip((2 * self.hp.max_abs_value) * ((S - self.hp.min_level_db) / (-self.hp.min_level_db)) - self.hp.max_abs_value,
                            -self.hp.max_abs_value, self.hp.max_abs_value)
            else:
                return np.clip(self.hp.max_abs_value * ((S - self.hp.min_level_db) / (-self.hp.min_level_db)), 0, self.hp.max_abs_value)
        
        assert S.max() <= 0 and S.min() - self.hp.min_level_db >= 0
        if self.hp.symmetric_mels:
            return (2 * self.hp.max_abs_value) * ((S - self.hp.min_level_db) / (-self.hp.min_level_db)) - self.hp.max_abs_value
        else:
            return self.hp.max_abs_value * ((S - self.hp.min_level_db) / (-self.hp.min_level_db))

    def _melspectrogram(self, wav: np.ndarray):
        D = self._stft(self._preemphasis(wav, self.hp.preemphasis, self.hp.preemphasize))
        S = self._amp_to_db(self._linear_to_mel(np.abs(D))) - self.hp.ref_level_db

        if self.hp.signal_normalization:
            return self._normalize(S)
        return S

    def _generate_blink_seq_randomly(self, num_frames: int):
        ratio = np.zeros((num_frames,1))
        if num_frames<=20:
            return ratio
        frame_id = 0
        while frame_id in range(num_frames):
            start = random.choice(range(min(10,num_frames), min(int(num_frames/2), 70))) 
            if frame_id+start+5<=num_frames - 1:
                ratio[frame_id+start:frame_id+start+5, 0] = [0.5, 0.9, 1.0, 0.9, 0.5]
                frame_id = frame_id+start+5
            else:
                break
        return ratio

    def _using_refpose(self, coeffs_pred_numpy, ref_pose_coeff_path):
        num_frames = coeffs_pred_numpy.shape[0]
        refpose_coeff_dict = scio.loadmat(ref_pose_coeff_path)
        refpose_coeff = refpose_coeff_dict['coeff_3dmm'][:,64:70]
        refpose_num_frames = refpose_coeff.shape[0]
        if refpose_num_frames<num_frames:
            div = num_frames//refpose_num_frames
            re = num_frames%refpose_num_frames
            refpose_coeff_list = [refpose_coeff for i in range(div)]
            refpose_coeff_list.append(refpose_coeff[:re, :])
            refpose_coeff = np.concatenate(refpose_coeff_list, axis=0)

        #### relative head pose
        coeffs_pred_numpy[:, 64:70] = coeffs_pred_numpy[:, 64:70] + ( refpose_coeff[:num_frames, :] - refpose_coeff[0:1, :] )
        return coeffs_pred_numpy

    def preprocess(
        self,
        first_coeff_path: str,
        audio_path: str,
        ref_eyeblink_coeff_path: Optional[str]=None,
        still: bool=False,
    ) -> dict:
        """build input data for AudioProcessor

        Args:
            first_coeff_path (str):  Path to the first coeff file
            audio_path (str): Path to the audio file
            device (str):  Device to use. e.g. "cpu", "cuda:0"
            ref_eyeblink_coeff_path Optional[str]:  Path to the reference eyeblink coeff file
            still (bool): Defaults to False.

        Returns:
            dict: Batch of data
        """        """"""
        syncnet_mel_step_size = 16
        fps = 25
        pic_name = os.path.splitext(os.path.split(first_coeff_path)[-1])[0]
        audio_name = os.path.splitext(os.path.split(audio_path)[-1])[0]

        # load audio file
        wav = self._load_wav(audio_path, 16000)
        wav_length, num_frames = self._parse_audio_length(len(wav), 16000, 25)
        wav = self._crop_pad_audio(wav, wav_length)

        # convert to melspectrogram
        orig_mel = self._melspectrogram(wav).T
        spec = orig_mel.copy()
        indiv_mels = []
        for i in tqdm(range(num_frames), 'mel:'):
            start_frame_num = i-2
            start_idx = int(80. * (start_frame_num / float(fps)))
            end_idx = start_idx + syncnet_mel_step_size
            seq = list(range(start_idx, end_idx))
            seq = [ min(max(item, 0), orig_mel.shape[0]-1) for item in seq ]
            m = spec[seq, :]
            indiv_mels.append(m.T)
        indiv_mels = np.asarray(indiv_mels, dtype=np.float32)
        indiv_mels = np.expand_dims(np.expand_dims(indiv_mels, axis=1), axis=0)

        ratio = self._generate_blink_seq_randomly(num_frames)      # T
        source_semantics_path = first_coeff_path
        source_semantics_dict = scio.loadmat(source_semantics_path)
        ref_coeff = source_semantics_dict['coeff_3dmm'][:1,:70]         #1 70
        ref_coeff = np.repeat(ref_coeff, num_frames, axis=0)
        ref_coeff = np.array(ref_coeff, dtype=np.float32)

        if ref_eyeblink_coeff_path is not None:
            raise Exception("ref_eyeblink_coeff_path is not supported yet")

        ref_coeff = np.expand_dims(ref_coeff, axis=0)
        
        if still:
            # ratio = torch.FloatTensor(ratio).unsqueeze(0).fill_(0.) # bs T
            ratio = np.array(np.expand_dims(ratio, axis=0), dtype=np.float32)
            ratio.fill(0.0)
        else:
            ratio = np.array(np.expand_dims(ratio, axis=0), dtype=np.float32)

        return {
            'indiv_mels': indiv_mels,  
            'ref': ref_coeff, 
            'num_frames': num_frames, 
            'ratio_gt': ratio,
            'audio_name': audio_name,
            'pic_name': pic_name,
        }

    def generate(
        self,
        batch,
        save_dir: str,
        pose_style: int,
        ref_pose_coeff_path: Optional[str] = None,
    ):
        batch['class'] = np.array([pose_style], dtype=np.int64)
        exp_pred = self.audio2exp_model.forward(batch['indiv_mels'], batch['ref'],  batch['ratio_gt'])
        results_dict_pose = self.audio_to_pose.forward(batch['ref'], batch['class'], batch['indiv_mels'], batch['num_frames'])
        pose_pred = results_dict_pose['pose_pred'] # bs T 6
        pose_len = pose_pred.shape[1]
        if pose_len < 13: 
            pose_len = int((pose_len-1)/2)*2+1
            pose_pred = savgol_filter(np.array(pose_pred), pose_len, 2, axis=1)
        else:
            pose_pred = savgol_filter(np.array(pose_pred), 13, 2, axis=1)

        coeffs_pred = np.concatenate([exp_pred, pose_pred], axis=-1)[0]
        if ref_pose_coeff_path is not None: 
            coeffs_pred = self._using_refpose(coeffs_pred, ref_pose_coeff_path)
        save_path = os.path.join(save_dir, '%s##%s.mat'%(batch['pic_name'], batch['audio_name']))
        scio.savemat(
            save_path,  
            {'coeff_3dmm': coeffs_pred},
        )
        return save_path

class AudioToExp:
    def __init__(self):
        self.model = onnxruntime.InferenceSession(AUDIO2EXP_PATH, providers=["CPUExecutionProvider"])

    def forward(self, mel_input:np.ndarray, ref: np.ndarray, ratio: np.ndarray):
        exp_pred = self.model.run(
            None,
            {
                "mel_input": mel_input,
                "ref": ref,
                "ratio": ratio,
            }
        )[0]
        return exp_pred

class AudioToPose:
    def __init__(self):
        self.audio_encoder = onnxruntime.InferenceSession(AUDIO2POSE_ENCODER_PATH, providers=["CPUExecutionProvider"])
        self.netG = onnxruntime.InferenceSession(AUDIO2POSE_NETG_PATH, providers=["CPUExecutionProvider"])

        self.seq_len = 32
        self.latent_dim = 64
        pass

    def forward(
        self,
        ref,
        _class,
        indiv_mels,
        num_frames,
    ):
        bs = ref.shape[0]
        batch = {}
        batch['ref'] = ref[:,0,-6:]
        batch['class'] = _class
        indiv_mels_use = indiv_mels[:, 1:]        # we regard the ref as the first frame
        num_frames = int(num_frames) - 1
        div = num_frames//self.seq_len
        re = num_frames%self.seq_len
        pose_motion_pred_list = [
            np.zeros(
                np.expand_dims(batch['ref'], axis=1).shape, 
                dtype=batch['ref'].dtype
            )
        ]
        for i in range(div):
            z = np.array(np.random.randn(bs, self.latent_dim), dtype=np.float32)
            batch['z'] = z
            result = self.audio_encoder.run(
                None,
                {
                    "melspectrograms": indiv_mels_use[:, i*self.seq_len:(i+1)*self.seq_len,:,:,:],
                },
            )
            batch['audio_emb'] = result[0]

            result = self.netG.run(
                None,
                {
                    "z": batch['z'],
                    "class": batch['class'],
                    "ref": batch["ref"],
                    "audio_emb": batch["audio_emb"],
                },
            )
            batch['pose_motion_pred'] = result[0]
            pose_motion_pred_list.append(batch['pose_motion_pred'])  #list of bs seq_len 6

        if re != 0:
            z = np.array(np.random.randn(bs, self.latent_dim), dtype=np.float32)
            batch['z'] = z
            result = self.audio_encoder.run(
                None,
                {
                    "melspectrograms": indiv_mels_use[:, -1*self.seq_len:,:,:,:],
                },
            )
            audio_emb = result[0]
            if audio_emb.shape[1] != self.seq_len:
                pad_dim = self.seq_len-audio_emb.shape[1]
                pad_audio_emb = audio_emb[:, :1].repeat(1, pad_dim, 1)
                # audio_emb = torch.cat([pad_audio_emb, audio_emb], 1)
                audio_emb = np.concatenate((pad_audio_emb, audio_emb), axis=1)
            batch['audio_emb'] = audio_emb
            result = self.netG.run(
                None,
                {
                    "z": batch['z'],
                    "class": batch['class'],
                    "ref": batch["ref"],
                    "audio_emb": batch["audio_emb"],
                },
            )
            batch['pose_motion_pred'] = result[0]
            pose_motion_pred_list.append(batch['pose_motion_pred'][:,-1*re:,:])

        pose_motion_pred = np.concatenate(pose_motion_pred_list, axis=1)
        batch['pose_motion_pred'] = pose_motion_pred
        pose_pred = ref[:, :1, -6:] + pose_motion_pred  # bs T 6
        batch['pose_pred'] = pose_pred
        return batch
