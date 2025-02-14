import numpy as np
import onnxruntime

class Audio2Pose:
    def __init__(self):
        self.audio2pose = onnxruntime.InferenceSession("./onnx/audio2pose.onnx")
        self.seq_len = 32
        self.latent_dim = 64

    def test(self, x):
        """
        元のtest()を修正した
            - AudioEncoderとnetGを1箇所にまとめる
            - これらを1つのonnxにエクスポートする
        """
        batch = {}

        # 1) 入力データの取得
        ref = x['ref'].numpy()  # [BS, 1, 70]
        class_id = x['class'].numpy()  # [BS]
        bs = ref.shape[0]
        pose_ref = ref[:, 0, -6:]  # [BS, 6]

        indiv_mels = x['indiv_mels'].numpy()  # [BS, T, 1, 80, 16]
        T_total = int(x['num_frames']) - 1  # refフレームを1つ使うので -1

        # 2) seq_len ごとにチャンク分割
        chunk_count = (T_total + self.seq_len - 1) // self.seq_len

        # 返却用リスト。最初に「pose_ref と同じ形の 0 フレーム」を1つ入れる
        pose_motion_pred_list = [
            np.zeros((bs, 1, pose_ref.shape[1]), dtype=np.float32)
        ]

        # 3) チャンクごとにループ処理
        start_idx = 0
        for i in range(chunk_count):
            # 3-1) 処理するフレーム範囲を決定
            end_idx = min(start_idx + self.seq_len, T_total)
            chunk_len = end_idx - start_idx

            # indiv_mels の該当部分を切り出し
            chunk_mels = indiv_mels[:, 1+start_idx : 1+end_idx]  # 先頭を ref フレームとしているなら +1

            # もし chunk_len < self.seq_len ならパディング
            if chunk_len < self.seq_len:
                pad_len = self.seq_len - chunk_len
                pad_chunk = np.repeat(chunk_mels[:, :1], pad_len, axis=1)
                chunk_mels = np.concatenate([pad_chunk, chunk_mels], axis=1)  

            # 3-2) ランダムノイズ生成 (潜在変数 z)
            z = np.random.randn(bs, self.latent_dim).astype(np.float32)

            # 3-3) AudioEncoder → netG
            motion_pred = self.audio2pose.run(None, {
                "chunk_mels": chunk_mels,
                "z": z,
                "pose_ref": pose_ref,
                "class": class_id,
            })[0]

            # 3-4) パディングした分は使わない
            if chunk_len < self.seq_len:
                motion_pred = motion_pred[:, -chunk_len:, :]

            # 結果をリストに追加
            pose_motion_pred_list.append(motion_pred)

            # 次のチャンク開始位置
            start_idx += chunk_len

        # 4) すべてのチャンクを結合
        pose_motion_pred = np.concatenate(pose_motion_pred_list, axis=1)  # [BS, T_total, 6]

        # 5) 累積ポーズ計算
        pose_pred = ref[:, :1, -6:] + pose_motion_pred  # [BS, T_total+1, 6]

        # 6) 結果を辞書に格納して返す
        batch['pose_motion_pred'] = pose_motion_pred
        batch['pose_pred'] = pose_pred
        return batch