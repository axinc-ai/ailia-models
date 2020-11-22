import ailia
from PIL import Image
import torch
import torchvision.transforms as transforms


class FaceParser:
    def __init__(self, device="cpu", args=None, face_parser_path=None):
        mapper = [0, 1, 2, 3, 4, 5, 0, 11, 12, 0, 6, 8, 7, 9, 13, 0, 0, 10, 0]
        self.device = device
        self.dic = torch.tensor(mapper, device=device)
        self.net = _initialize_net(args, face_parser_path)
        self.args = args
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def parse(self, image: Image):
        assert image.shape[:2] == (512, 512)
        with torch.no_grad():
            image = self.to_tensor(image).to(self.device)
            image = torch.unsqueeze(image, 0)
            out = _parse_face(image, self.net, self.args)[0]
            parsing = torch.from_numpy(out.squeeze(0).argmax(0))
        parsing = torch.nn.functional.embedding(parsing, self.dic)
        return parsing.float()


def _initialize_net(args, face_parser_path):
    env_id = ailia.get_gpu_environment_id()
    print(f"env_id (face parser): {env_id}")
    if not args.onnx:
        net = ailia.Net(face_parser_path[0], face_parser_path[1], env_id=env_id)
    else:
        import onnxruntime

        net = onnxruntime.InferenceSession(face_parser_path[1])

    return net


def _parse_face(input_image, net, args):
    if not args.onnx:
        return net.predict(_to_numpy(input_image))
    else:
        inputs = {net.get_inputs()[0].name: _to_numpy(input_image)}
        return net.run(None, inputs)


def _to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
