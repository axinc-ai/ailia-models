import onnx

def modify_onnx(model_path: str, output_path: str):
    model = onnx.load(model_path)

    modified = False

    # RandomUniformLike の dtype を 1→10 に変更
    for node in model.graph.node:
        if node.op_type == "RandomUniformLike" or node.op_type == "RandomUniform" or node.op_type == "RandomNormalLike":
            for attr in node.attribute:
                if attr.name == "dtype" and attr.type == onnx.AttributeProto.INT:
                    if attr.i == 1:
                        print(f"{node.op_type} ('{node.name}') dtype: {attr.i} -> 10")
                        attr.i = 10
                        modified = True

    # Cast の to 属性を Float(1) → Float16(10) に変更
    # onnxの Castオペレータでは "to" 属性(i型)が目的のdtype
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.type == onnx.AttributeProto.INT:
                    if attr.i == 1:  # float
                        print(f"Cast('{node.name}') to: {attr.i} -> 10")
                        attr.i = 10
                        modified = True

    if modified:
        onnx.save(model, output_path)
        print(f"Saved modified model to: {output_path}")
    else:
        print("No modifications were applied.")

if __name__ == "__main__":
    # 例: python modify_onnx_dtype.py input.onnx output.onnx
    import sys
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <input.onnx> <output.onnx>")
        sys.exit(1)

    input_model = sys.argv[1]
    output_model = sys.argv[2]

    modify_onnx(input_model, output_model)
