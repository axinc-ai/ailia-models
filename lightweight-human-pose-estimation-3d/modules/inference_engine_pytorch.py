import numpy as np
import torch


class InferenceEnginePyTorch:
    def __init__(self, checkpoint_path, device,
                 img_mean=np.array([128, 128, 128], dtype=np.float32),
                 img_scale=np.float32(1/255)):
        from models.with_mobilenet import PoseEstimationWithMobileNet
        from modules.load_state import load_state
        self.img_mean = img_mean
        self.img_scale = img_scale
        self.device = 'cpu'
        if device != 'CPU':
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            else:
                print('No CUDA device found, inferring on CPU')

        net = PoseEstimationWithMobileNet()
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        load_state(net, checkpoint)
        net = net.to(self.device)
        net.eval()
        self.net = net

    def infer(self, img):
        normalized_img = InferenceEnginePyTorch._normalize(img, self.img_mean, self.img_scale)
        data = torch.from_numpy(normalized_img).permute(2, 0, 1).unsqueeze(0).to(self.device)

        features, heatmaps, pafs = self.net(data)

        return (features[-1].squeeze().data.cpu().numpy(),
                heatmaps[-1].squeeze().data.cpu().numpy(), pafs[-1].squeeze().data.cpu().numpy())

    @staticmethod
    def _normalize(img, img_mean, img_scale):
        normalized_img = (img.astype(np.float32) - img_mean) * img_scale
        return normalized_img

