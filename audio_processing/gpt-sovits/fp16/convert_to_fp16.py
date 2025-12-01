import onnx
from onnxconverter_common.float16 import convert_float_to_float16

models = ["t2s_fsdec.onnx", "vits.onnx", "t2s_encoder.onnx", "cnhubert.onnx", "vits.onnx", "t2s_sdec.opt3.onnx", "t2s_fsdec.onnx"]

for model_name in models:
	model = onnx.load("../" + model_name)
	model_fp16 = convert_float_to_float16(model, disable_shape_infer=True, keep_io_types=False)
	onnx.save(model_fp16, model_name.replace(".onnx", "_fp16.onnx"))
