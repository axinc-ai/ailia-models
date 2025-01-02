import cv2
import rembg
import numpy as np
import trimesh

from skimage.measure import marching_cubes
from collections import defaultdict
from dataclasses import dataclass

def scale_tensor(dat, inp_scale, tgt_scale):
    if inp_scale is None:
        inp_scale = (0, 1)
    if tgt_scale is None:
        tgt_scale = (0, 1)
    dat = (dat - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat


def remove_background(
    image,
    rembg_session = None,
    force: bool = False,
    **rembg_kwargs):
    rembg_session = rembg.new_session()
    do_remove = True
    if image.shape[2] == 4 and image.min() < 255:
        do_remove = False
    do_remove = do_remove or force
    if do_remove:
        image = rembg.remove(image, session=rembg_session )
    return image


def resize_foreground(
    image,
    ratio: float):
    image = np.array(image)
    alpha = np.where(image[..., 3] > 0)
    y1, y2, x1, x2 = (
        alpha[0].min(),
        alpha[0].max(),
        alpha[1].min(),
        alpha[1].max(),
    )
    # crop the foreground
    fg = image[y1:y2, x1:x2]
    # pad to square
    size = max(fg.shape[0], fg.shape[1])
    ph0, pw0 = (size - fg.shape[0]) // 2, (size - fg.shape[1]) // 2
    ph1, pw1 = size - fg.shape[0] - ph0, size - fg.shape[1] - pw0
    new_image = np.pad(
        fg,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )

    # compute padding according to the ratio
    new_size = int(new_image.shape[0] / ratio)
    # pad to size, double side
    ph0, pw0 = (new_size - size) // 2, (new_size - size) // 2
    ph1, pw1 = new_size - size - ph0, new_size - size - pw0
    new_image = np.pad(
        new_image,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((0, 0), (0, 0), (0, 0)),
    )
    return new_image

def grid_sample_cv2(triplane, indices, align_corners=False):
    Np, Cp, Hp, Wp = triplane.shape
    _, Hout, Wout, _ = indices.shape
    
    if align_corners:
        x = ((indices[:, :, :, 0] + 1) * (Wp - 1) / 2).astype(np.float32)
        y = ((indices[:, :, :, 1] + 1) * (Hp - 1) / 2).astype(np.float32)
    else:
        x = ((indices[:, :, :, 0] + 1) * Wp / 2 - 0.5).astype(np.float32)
        y = ((indices[:, :, :, 1] + 1) * Hp / 2 - 0.5).astype(np.float32)

    output = np.zeros((Np, Cp, Hout, Wout), dtype=np.float32)
    
    for n in range(Np):
        for c in range(Cp):
            output[n, c] = cv2.remap(
                triplane[n, c], x[n], y[n], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
            )
    
    return output

class TriplaneNeRFRenderer():
    def __init__(self,decoder,radius,feature_reduction,density_activation,density_bias,num_samples_per_ray):
        self.decoder = decoder
        self.radius =  radius
        self.feature_reduction   = feature_reduction
        self.density_activation  = density_activation
        self.density_bias        = density_bias
        self.num_samples_per_ray = num_samples_per_ray

    def configure(self) -> None:
        assert self.feature_reduction in ["concat", "mean"]
        self.chunk_size = 0

    def set_chunk_size(self, chunk_size: int):
        assert (
            chunk_size >= 0
        ), "chunk_size must be a non-negative integer (0 for no chunking)."
        self.chunk_size = chunk_size

    def chunk_batch1(self,triplane, chunk_size: int, positions, **kwargs):
        if chunk_size <= 0:
            return func(*args, **kwargs)
        B = positions.shape[0]
        # max(1, B) to support B == 0
        
        output = []
        for i in range(0, max(1, B), chunk_size):

            chunk1 = [positions[i : i + chunk_size]]
            x = chunk1[0]

            indices2D = np.stack(
                (x[:, [0, 1]], x[:, [0, 2]], x[:, [1, 2]]),
                axis=-3,
            )

            triplane1 = np.einsum('nchw->nchw', triplane)

            indices2D = np.expand_dims(indices2D,axis=1)

            out1 = grid_sample_cv2(triplane1,indices2D,align_corners=False)
            if self.feature_reduction == "concat":
                Np, Cp, _, N = out1.shape

                out1 = np.squeeze(out1, axis=2)  # shape: (Np, Cp, N)
                out1 = np.transpose(out1, (2, 0, 1))  # shape: (N, Np, Cp)
                out1 = np.reshape(out1, (N, Np * Cp))  # shape: (N, Np * Cp)

            elif self.feature_reduction == "mean":
                Np, Cp, _, N = out1.shape

                out1 = np.squeeze(out1, axis=2)  # shape: (Np, Cp, N)
                out1 = np.transpose(out1, (2, 0, 1))  # shape: (N, Np, Cp)
                out1 = np.mean(out1,axis=1)  # shape: (N, Np * Cp)
                
            else:
                raise NotImplementedError
            output.append(out1)
        return output

    def chunk_batch2(self,inputs,chunk_size: int,pos:int):

        out = defaultdict(list)
        out_type = None
        #for input in inputs:
        B = pos.shape[0]
        for i in range(0, max(1, B), chunk_size):
            input = inputs[i//chunk_size]
            results = self.decoder.run(input)
        
            input =  (np.array(results[0]) ,np.array(results[1]))
            out_chunk = input
 
            if out_chunk is None:
                continue
            out_type = type(out_chunk)
            if isinstance(out_chunk, tuple) or isinstance(out_chunk, list):
                chunk_length = len(out_chunk)
                out_chunk = {i: chunk for i, chunk in enumerate(out_chunk)}
            elif isinstance(out_chunk, dict):
                pass
            for k, v in out_chunk.items():
                out[k].append(v)
    
        out_merged = {}
        for k, v in out.items():
            out_merged[k] = np.concatenate(v, axis=0)
            continue

        if out_type in [tuple, list]:
            return out_type([out_merged[i] for i in range(chunk_length)])
        elif out_type is dict:
            return out_merged

    def query_triplane(
        self,positions,triplane):

        input_shape = positions.shape[:-1]
        positions = positions.reshape(-1, 3)

        positions = scale_tensor(
            positions, (-self.radius, self.radius), (-1, 1)
        )

        if self.chunk_size > 0:
            net_out = self.chunk_batch1(triplane,self.chunk_size,positions)
            net_out = self.chunk_batch2(net_out,self.chunk_size,positions)
        
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        net_out_density_act =np.exp(
            net_out[0] + self.density_bias
        )
        net_out_color = sigmoid(
            net_out[1]
        )

        net_out = (net_out[0].reshape(*input_shape, -1), net_out[1].reshape(*input_shape, -1))
        net_out_density_act = net_out_density_act.reshape(*input_shape, -1)
        net_out_color = net_out_color.reshape(*input_shape, -1)

        return net_out[0],net_out[1], net_out_density_act,net_out_color



@dataclass
class Config():
    renderer_cls: str
    renderer: dict


class TSR():
    def __init__(self,
                decoder,
                radius,
                feature_reduction,
                density_activation,
                density_bias,
                num_samples_per_ray):
        self.cfg = Config("",{})
        self.cfg.renderer = {"radius":radius,
                             "feature_reduction":feature_reduction,
                             "density_activation":density_activation,
                             "density_bias":density_bias,
                             "num_samples_per_ray":num_samples_per_ray}

        self.configure(decoder)

    def configure(self,decoder):
        self.renderer =TriplaneNeRFRenderer(decoder,**self.cfg.renderer)
        self.isosurface_helper = None

    def extract_mesh2(self, scene_code, resolution: int = 256, threshold: float = 25.0,device = 'cpu'):
        points_range = (0,1)
        x, y, z = (
            np.linspace(*points_range, resolution),
            np.linspace(*points_range, resolution),
            np.linspace(*points_range, resolution),
        )
        x, y, z = np.meshgrid(x, y, z, indexing="ij")
        verts = np.concatenate(
            [x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], axis=1
        ).reshape(-1, 3)


        density = self.renderer.query_triplane(
            scale_tensor(
                verts,
                (0,1),
               (-self.renderer.radius, self.renderer.radius),
            ),
            scene_code,
        )[2]
 
        level = -(density - threshold)

        level = -level.reshape(resolution, resolution, resolution)

        v_pos, t_pos_idx,_,_= marching_cubes(level.copy(), 0.0)
        v_pos = v_pos / (resolution - 1.0)
        t_pos_idx = t_pos_idx.copy()


        v_pos = scale_tensor( v_pos,
            (0,1),
            (-self.renderer.radius, self.renderer.radius),
        )

        v_pos = v_pos.copy()
        color = self.renderer.query_triplane(
            v_pos,
            scene_code,
        )[3]
        return v_pos, t_pos_idx,color,
     
 
    def extract_mesh(self, scene_codes, resolution: int = 256, threshold: float = 25.0):
        meshes = []
        for scene_code in scene_codes:
            v_pos , t_pos_idx,color = self.extract_mesh2(scene_code,resolution,threshold)

            mesh = trimesh.Trimesh(
                vertices=v_pos,
                faces=t_pos_idx,
                vertex_colors=color,
            ) 
            meshes.append(mesh)
        return meshes

