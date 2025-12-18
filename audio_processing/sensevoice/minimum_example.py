# minimum infer example

import soundfile
import librosa

import numpy as np

# -*- encoding: utf-8 -*-
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np

import librosa
import ailia.audio

from pathlib import Path
from typing import List
from typing import Union

import os.path
import librosa
import numpy as np
from pathlib import Path
from typing import List, Union, Tuple
import yaml

frame_opts_samp_freq = 16000
mel_opts_num_bins = 80
frame_opts_window_type = "hamming"
frame_opts_frame_length_ms = 25
frame_opts_frame_shift_ms = 10

class SimpleDecoder():
	def accept_waveform(self, sampling_rate: float, waveform):
		self.sr = int(frame_opts_samp_freq)
		self.dither = 1.0
		self.window_type = frame_opts_window_type
		self.frame_length_ms = float(frame_opts_frame_length_ms)
		self.frame_shift_ms = float(frame_opts_frame_shift_ms)
		self.n_mels = int(mel_opts_num_bins)
		self.frame_length = int(self.sr * self.frame_length_ms / 1000)
		self.frame_shift = int(self.sr * self.frame_shift_ms / 1000)
		self.snip_edges = True
		self._waveform_buffer = np.array([], dtype=np.float32)
		self._frames_cache = None
		self.preemphasis_coefficient = 0.97
			
		waveform = np.asarray(waveform, dtype=np.float32)
		if self.dither > 0:
			waveform += np.random.normal(scale=self.dither, size=len(waveform)).astype(np.float32)
		if self.preemphasis_coefficient > 0: # Kaldiの内部処理をシミュレート
			alpha = self.preemphasis_coefficient
			waveform = np.append(waveform[0], waveform[1:] - alpha * waveform[:-1])
		self._waveform_buffer = np.concatenate([self._waveform_buffer, waveform])

		"""Compute mel filterbank features for all frames available."""
		if len(self._waveform_buffer) < self.frame_length:
			self._frames_cache = np.empty((0, self.n_mels), dtype=np.float32)
			return

		hop_length = self.frame_shift
		win_length = self.frame_length

		# librosa uses float[-1, 1] range, assume waveform already scaled appropriately
		mel_spec = ailia.audio.mel_spectrogram(
			wav=self._waveform_buffer,
			sample_rate=self.sr,
			fft_n= win_length,
			hop_n=hop_length,
			win_n=win_length,
			win_type=self.window_type,
			mel_n=self.n_mels,
			center_mode=not self.snip_edges,
			power=2.0,
		)
		mel_spec = np.log(np.maximum(mel_spec, 1e-10)).T.astype(np.float32)
		self._frames_cache = mel_spec

	@property
	def num_frames_ready(self):
		return 0 if self._frames_cache is None else self._frames_cache.shape[0]

	def get_frame(self, i):
		"""Get i-th fbank frame as numpy array."""
		if self._frames_cache is None or i >= self._frames_cache.shape[0]:
			raise IndexError("Frame index out of range or cache empty.")
		return self._frames_cache[i]

	def get_all_frames(self):
		"""Return all computed frames."""
		return self._frames_cache

	def fbank(self, waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		waveform = waveform * (1 << 15)
		self.accept_waveform(frame_opts_samp_freq, waveform.tolist())
		frames = self.num_frames_ready
		mat = np.empty([frames, mel_opts_num_bins])
		for i in range(frames):
			mat[i, :] = self.get_frame(i)
		feat = mat.astype(np.float32)
		feat_len = np.array(mat.shape[0]).astype(np.int32)
		return feat, feat_len

	def lfr_cmvn(self, feat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		if self.lfr_m != 1 or self.lfr_n != 1:
			feat = self.apply_lfr(feat, self.lfr_m, self.lfr_n)

		if self.cmvn_file:
			feat = self.apply_cmvn(feat)

		feat_len = np.array(feat.shape[0]).astype(np.int32)
		return feat, feat_len

	@staticmethod
	def apply_lfr(inputs: np.ndarray, lfr_m: int, lfr_n: int) -> np.ndarray:
		LFR_inputs = []

		T = inputs.shape[0]
		T_lfr = int(np.ceil(T / lfr_n))
		left_padding = np.tile(inputs[0], ((lfr_m - 1) // 2, 1))
		inputs = np.vstack((left_padding, inputs))
		T = T + (lfr_m - 1) // 2
		for i in range(T_lfr):
			if lfr_m <= T - i * lfr_n:
				LFR_inputs.append((inputs[i * lfr_n : i * lfr_n + lfr_m]).reshape(1, -1))
			else:
				# process last LFR frame
				num_padding = lfr_m - (T - i * lfr_n)
				frame = inputs[i * lfr_n :].reshape(-1)
				for _ in range(num_padding):
					frame = np.hstack((frame, inputs[-1]))

				LFR_inputs.append(frame)
		LFR_outputs = np.vstack(LFR_inputs).astype(np.float32)
		return LFR_outputs

	def apply_cmvn(self, inputs: np.ndarray) -> np.ndarray:
		"""
		Apply CMVN with mvn data
		"""
		frame, dim = inputs.shape
		means = np.tile(self.cmvn[0:1, :dim], (frame, 1))
		vars = np.tile(self.cmvn[1:2, :dim], (frame, 1))
		inputs = (inputs + means) * vars
		return inputs

	def load_cmvn(cmvn_file: Union[str, Path]) -> np.ndarray:
		"""load cmvn file to numpy array. 

		Args:
			cmvn_file (Union[str, Path]): cmvn file path.

		Raises:
			FileNotFoundError: cmvn file not exits.

		Returns:
			np.ndarray: cmvn array. shape is (2, dim).The first row is means, the second row is vars.
		"""

		cmvn_file = Path(cmvn_file)
		if not cmvn_file.exists():
			raise FileNotFoundError("cmvn file not exits")
		
		with open(cmvn_file, "r", encoding="utf-8") as f:
			lines = f.readlines()
		means_list = []
		vars_list = []
		for i in range(len(lines)):
			line_item = lines[i].split()
			if line_item[0] == "<AddShift>":
				line_item = lines[i + 1].split()
				if line_item[0] == "<LearnRateCoef>":
					add_shift_line = line_item[3 : (len(line_item) - 1)]
					means_list = list(add_shift_line)
					continue
			elif line_item[0] == "<Rescale>":
				line_item = lines[i + 1].split()
				if line_item[0] == "<LearnRateCoef>":
					rescale_line = line_item[3 : (len(line_item) - 1)]
					vars_list = list(rescale_line)
					continue

		means = np.array(means_list).astype(np.float64)
		vars = np.array(vars_list).astype(np.float64)
		cmvn = np.array([means, vars])
		return cmvn

	def read_yaml(yaml_path: Union[str, Path]) -> Dict:
		if not Path(yaml_path).exists():
			raise FileExistsError(f"The {yaml_path} does not exist.")

		with open(str(yaml_path), "rb") as f:
			data = yaml.load(f, Loader=yaml.Loader)
		return data


	def _get_lid(self, lid):
		if lid in list(self.lid_dict.keys()):
			return self.lid_dict[lid]
		else:
			raise ValueError(
				f"The language {lid} is not in {list(self.lid_dict.keys())}"
			)
			
	def _get_tnid(self, tnid):
		if tnid in list(self.textnorm_dict.keys()):
			return self.textnorm_dict[tnid]
		else:
			raise ValueError(
				f"The textnorm {tnid} is not in {list(self.textnorm_dict.keys())}"
			)
	
	def read_tags(self, language_input, textnorm_input):
		# handle language
		if isinstance(language_input, list):
			language_list = []
			for l in language_input:
				language_list.append(self._get_lid(l))
		elif isinstance(language_input, str):
			# if is existing file
			if os.path.exists(language_input):
				language_file = open(language_input, "r").readlines()
				language_list = [
					self._get_lid(l.strip())
					for l in language_file
				]
			else:
				language_list = [self._get_lid(language_input)]
		else:
			raise ValueError(
				f"Unsupported type {type(language_input)} for language_input"
			)
		# handle textnorm
		if isinstance(textnorm_input, list):
			textnorm_list = []
			for tn in textnorm_input:
				textnorm_list.append(self._get_tnid(tn))
		elif isinstance(textnorm_input, str):
			# if is existing file
			if os.path.exists(textnorm_input):
				textnorm_file = open(textnorm_input, "r").readlines()
				textnorm_list = [
					self._get_tnid(tn.strip())
					for tn in textnorm_file
				]
			else:
				textnorm_list = [self._get_tnid(textnorm_input)]
		else:
			raise ValueError(
				f"Unsupported type {type(textnorm_input)} for textnorm_input"
			)
		return language_list, textnorm_list

	def __call__(self, wav_content: Union[str, np.ndarray, List[str]], **kwargs):
		import ailia
		self.model = ailia.Net(weight="sensevoice_small.onnx", env_id=1, memory_mode=11)

		from ailia_tokenizer import LlamaTokenizer
		self.tokenizer = LlamaTokenizer.from_pretrained("./tokenizer")

		config_file = "./s2t_config/config.yaml"
		cmvn_file ="./s2t_config/am.mvn"

		self.config = self.read_yaml(config_file)
		self.cmvn = self.load_cmvn(cmvn_file)

		self.batch_size = 1
		self.blank_id = 0
		self.lid_dict = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
		self.lid_int_dict = {24884: 3, 24885: 4, 24888: 7, 24892: 11, 24896: 12, 24992: 13}
		self.textnorm_dict = {"withitn": 14, "woitn": 15}
		self.textnorm_int_dict = {25016: 14, 25017: 15}

		language_input = kwargs.get("language", "auto")
		textnorm_input = kwargs.get("textnorm", "woitn")
		language_list, textnorm_list = self.read_tags(language_input, textnorm_input)
		
		waveform_list = [wav_content]
		waveform_nums = len(waveform_list)
		
		assert len(language_list) == 1 or len(language_list) == waveform_nums, \
			"length of parsed language list should be 1 or equal to the number of waveforms"
		assert len(textnorm_list) == 1 or len(textnorm_list) == waveform_nums, \
			"length of parsed textnorm list should be 1 or equal to the number of waveforms"
		
		asr_res = []
		for beg_idx in range(0, waveform_nums, self.batch_size):
			end_idx = min(waveform_nums, beg_idx + self.batch_size)
			feats, feats_len = self.extract_feat(waveform_list[beg_idx:end_idx])
			_language_list = language_list[beg_idx:end_idx]
			_textnorm_list = textnorm_list[beg_idx:end_idx]
			if not len(_language_list):
				_language_list = [language_list[0]]
				_textnorm_list = [textnorm_list[0]]
			B = feats.shape[0]
			if len(_language_list) == 1 and B != 1:
				_language_list = _language_list * B
			if len(_textnorm_list) == 1 and B != 1:
				_textnorm_list = _textnorm_list * B
			ctc_logits, encoder_out_lens = self.infer(
				feats,
				feats_len,
				np.array(_language_list, dtype=np.int32),
				np.array(_textnorm_list, dtype=np.int32),
			)
			for b in range(feats.shape[0]):
				# back to torch.Tensor
				# if isinstance(ctc_logits, np.ndarray):
				#     ctc_logits = torch.from_numpy(ctc_logits).float()
				# support batch_size=1 only currently
				x = ctc_logits[b, : encoder_out_lens[b].item(), :]
				yseq = np.argmax(x, axis=-1)
				# Use np.diff and np.where instead of torch.unique_consecutive.
				mask = np.concatenate(([True], np.diff(yseq) != 0))
				yseq = yseq[mask]

				mask = yseq != self.blank_id
				token_int = yseq[mask].tolist()

				text = self.tokenizer.decode(token_int, skip_special_tokens=True)
				asr_res.append(text)

		return asr_res

	def extract_feat(self, waveform_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
		feats, feats_len = [], []
		for waveform in waveform_list:
			speech, _ = self.frontend.fbank(waveform)

			if speech is None or speech.size == 0:
				print("detected speech size {speech.size}")
				raise ValueError("Empty speech detected, skipping this waveform.")
			feat, feat_len = self.frontend.lfr_cmvn(speech)
			feats.append(feat)
			feats_len.append(feat_len)

		feats = self.pad_feats(feats, np.max(feats_len))
		feats_len = np.array(feats_len).astype(np.int32)
		return feats, feats_len

	@staticmethod
	def pad_feats(feats: List[np.ndarray], max_feat_len: int) -> np.ndarray:
		def pad_feat(feat: np.ndarray, cur_len: int) -> np.ndarray:
			pad_width = ((0, max_feat_len - cur_len), (0, 0))
			return np.pad(feat, pad_width, "constant", constant_values=0)

		feat_res = [pad_feat(feat, feat.shape[0]) for feat in feats]
		feats = np.array(feat_res).astype(np.float32)
		return feats

	def infer(
		self,
		feats: np.ndarray,
		feats_len: np.ndarray,
		language: np.ndarray,
		textnorm: np.ndarray,
	) -> Tuple[np.ndarray, np.ndarray]:
		outputs = self.ort_infer([feats, feats_len, language, textnorm])
		return outputs


def main():
	speech, sample_rate = soundfile.read("ja.wav")
	if speech.ndim > 1:
		speech = np.mean(speech, axis=1)
	target_sr = 16000
	if sample_rate != target_sr:
		speech = librosa.resample(speech, orig_sr=sample_rate, target_sr=target_sr)

	wav_or_scp = speech
	decoder = SimpleDecoder()
	res = decoder(wav_or_scp, language="auto", use_itn=True) # 16khz
	print(res)

if __name__ == "__main__":
	main()
