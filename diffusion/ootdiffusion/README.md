# OOTDiffusion

## Input

![Input model image](model.png)

(Image from ...)

![Input cloth image](cloth.jpg)

(Image from ...)

## Output

TBD

## Usage
Automatically downloads the onnx and prototxt files on the first run.
It is necessary to be connected to the Internet while downloading.

For the sample video,
``` bash
$ python3 ootdiffusion.py
```

By adding the `--onnx` option, you can run the inference using OnnxRuntime.
```bash
$ python3 ootdiffusion.py --onnx
```

## Reference

- [OOTDiffusion](https://github.com/levihsu/OOTDiffusion)

## Framework

Pytorch

## Model Format

ONNX opset = 14

Models can be downloaded from [ax Drive](https://drive.google.com/file/d/1gsdKNyILaNfVxI-8lb3fIoHHJh5KO3GW/view?usp=sharing).

## Problems

- The models `unet_garm_hd.onnx` and `unet_garm_dc.onnx` were successfully exported from PyTorch to ONNX and loaded by both ailia SDK and ONNXRuntime, but get killed on inference (OOM). <br>
Basic example:
```python
import numpy as np

WEIGHT_UNET_GARM_HD_PATH = 'unet_garm_hd.onnx'

garm_latents = np.random.rand(2, 4, 128, 96).astype(np.float16)
prompt_embeds = np.random.rand(2, 2, 768).astype(np.float16)
timestep = np.array([0], dtype=np.int32)


# Using ailia SDK
import ailia

memory_mode = ailia.get_memory_mode(True, True, True, True)
unet_garm = ailia.Net(None, WEIGHT_UNET_GARM_HD_PATH, memory_mode=memory_mode)
unet_garm.run([garm_latents, timestep, prompt_embeds])  # Gets killed (OOM)


# Using ORT
import onnxruntime as ort

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
unet_garm = ort.InferenceSession(WEIGHT_UNET_GARM_HD_PATH, providers=providers)
unet_garm.run(None, {unet_garm.get_inputs()[0].name: garm_latents,
                        unet_garm.get_inputs()[1].name: timestep,
                        unet_garm.get_inputs()[2].name: prompt_embeds})  # Gets killed (OOM)
```

- The models `unet_vton_hd.onnx` and `unet_vton_dc.onnx` were successfully exported from PyTorch to ONNX and loaded by both ailia SDK and ONNXRuntime, but get killed on inference (OOM). <br>
Basic example:
```python
import numpy as np

WEIGHT_UNET_VTON_HD_PATH = 'unet_vton_hd.onnx'

all_inputs_list = [
    np.random.randn(2, 8, 128, 96).astype(np.float16),
    np.random.randn(2, 12288, 320).astype(np.float16),
    np.random.randn(2, 12288, 320).astype(np.float16),
    np.random.randn(2, 3072, 640).astype(np.float16),
    np.random.randn(2, 3072, 640).astype(np.float16),
    np.random.randn(2, 768, 1280).astype(np.float16),
    np.random.randn(2, 768, 1280).astype(np.float16),
    np.random.randn(2, 192, 1280).astype(np.float16),
    np.random.randn(2, 768, 1280).astype(np.float16),
    np.random.randn(2, 768, 1280).astype(np.float16),
    np.random.randn(2, 768, 1280).astype(np.float16),
    np.random.randn(2, 3072, 640).astype(np.float16),
    np.random.randn(2, 3072, 640).astype(np.float16),
    np.random.randn(2, 3072, 640).astype(np.float16),
    np.random.randn(2, 12288, 320).astype(np.float16),
    np.random.randn(2, 12288, 320).astype(np.float16),
    np.random.randn(2, 12288, 320).astype(np.float16),
    np.array([0]).astype(np.int32),
    np.random.randn(2, 4, 768).astype(np.float16)
]

all_inputs_dict = {'sample': all_inputs_list[0],
                    'timestep': all_inputs_list[-2],
                    'encoder_hidden_states': all_inputs_list[-1]}

for i, value in enumerate(all_inputs_list[1:-2]):
    all_inputs_dict[f'spatial_attn_inputs_{i}'] = value


# Using ailia SDK
import ailia

memory_mode = ailia.get_memory_mode(True, True, True, True)
unet_vton = ailia.Net(None, WEIGHT_UNET_VTON_HD_PATH, memory_mode=memory_mode)
unet_vton.run(all_inputs_list)[0]  # Gets killed
# Error details: Floating point exception (core dumped)


# Using ORT
import onnxruntime as ort

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
unet_vton = ort.InferenceSession(WEIGHT_UNET_VTON_HD_PATH, providers=providers)
unet_vton.run(None, all_inputs_dict)[0]  # Gets killed
# Error details:
# [ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Non-zero status code returned while running Softmax node. Name:'/down_blocks.0/attentions.0/transformer_blocks.0/attn1/Softmax' Status Message: /onnxruntime_src/onnxruntime/core/framework/bfc_arena.cc:376 void* onnxruntime::BFCArena::AllocateRawInternal(size_t, bool, onnxruntime::Stream*, bool, onnxruntime::WaitNotificationFn) Failed to allocate memory for requested buffer of size 19327352832
```

## Netron
