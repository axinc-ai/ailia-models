from tqdm import tqdm
import numpy as np
import onnxruntime

class Audio2Exp:
    def __init__(self):
        self.netG = onnxruntime.InferenceSession("./onnx/audio2exp.onnx")

    def test(self, batch):
        mel_input = batch['indiv_mels'].numpy()  # (bs, T, 1, 80, 16)
        ref = batch['ref'].numpy()              # (bs, T, 70)
        ratio = batch['ratio_gt'].numpy()       # (bs, T, 1)

        bs, T, _, _, _ = mel_input.shape
        exp_coeff_pred = []

        for i in tqdm(range(0, T, 10), 'audio2exp:'):  # 10フレームごとに処理
            current_mel_input = mel_input[:, i:i+10]  # (bs, 10, 1, 80, 16)
            current_ref = ref[:, i:i+10, :64]         # (bs, 10, 64)
            current_ratio = ratio[:, i:i+10]         # (bs, 10, 1)

            audiox = current_mel_input.reshape(-1, 1, 80, 16)  # (bs*T, 1, 80, 16)

            curr_exp_coeff_pred = self.netG.run(None, {
                "audio": audiox,
                "ref": current_ref,
                "ratio": current_ratio,
            })[0]

            exp_coeff_pred += [curr_exp_coeff_pred]

        # BS x T x 64
        results_dict = {
            'exp_coeff_pred': np.concatenate(exp_coeff_pred, axis=1)
        }

        return results_dict
