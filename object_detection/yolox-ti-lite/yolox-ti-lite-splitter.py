import numpy             # to modify BatchNormalization/Conv parameter
import onnx              # to load/save ONNX model file
import onnx.helper       # to update ONNX model
import onnx.numpy_helper # to convert ONNX TensorProto <-> numpy ndarray

# input/output filename
src_onnx = 'yolox-s-ti-lite_39p1_57p9.onnx' 
opt_onnx = 'yolox-s-ti-lite_39p1_57p9.opt.onnx'
opt2_onnx = 'yolox-s-ti-lite_39p1_57p9.opt2.onnx'

def onnx_split( model ) :
    mod_initializer = []
    for e in model.graph.initializer :
        mod_initializer.append( e )

    mod_node = []
    mod_output = []
    skip = False
    for n in model.graph.node :
        if not skip:
            mod_node.append(n)
        if n.name=="Transpose_583":
            skip=True
            mod_output = [onnx.helper.make_tensor_value_info("detections", onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[numpy.dtype('float32')], (1, 8400, 85))]
            n.output[0] = mod_output[0].name

    # generate modified model
    m_graph = onnx.helper.make_graph(mod_node, model.graph.name, model.graph.input, mod_output, mod_initializer)
    m_model = onnx.helper.make_model(m_graph, producer_name=model.producer_name, producer_version=model.producer_version, opset_imports=model.opset_import)

    return m_model

# load model
model = onnx.load(src_onnx)

# split onnx
model = onnx_split( model )

# save optimized model
with open(opt_onnx, "wb") as f:
    f.write(model.SerializeToString())

# fix shape for TIDL
import onnx
onnx.shape_inference.infer_shapes_path(opt_onnx, opt2_onnx)
