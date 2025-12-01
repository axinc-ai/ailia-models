# FP32のWhisperをFP16に変換する

# onnxconverter-common : 1.16.0
# onnx : 1.19.1

import onnx
from onnxconverter_common.float16 import convert_float_to_float16
arch = "tiny"

op_block_list = [
    "ArrayFeatureExtractor",
    "Binarizer",
    "CastMap",
    "CategoryMapper",
    "DictVectorizer",
    "FeatureVectorizer",
    "Imputer",
    "LabelEncoder",
    "LinearClassifier",
    "LinearRegressor",
    "Normalizer",
    "OneHotEncoder",
    "RandomUniformLike",
    "SVMClassifier",
    "SVMRegressor",
    "Scaler",
    "TreeEnsembleClassifier",
    "TreeEnsembleRegressor",
    "ZipMap",
    "NonMaxSuppression",
    "TopK",
    "RoiAlign",
    "Resize",
    "Range",
    "CumSum",
    "Min",
    "Max",
    "Upsample",
    "Pow" # Added (これがないとFP16に変換後、onnxでFP32に戻されて、initializer = FP16と不一致のエラーになる)
]

model = onnx.load("../encoder_"+arch+".opt3.onnx")
model_fp16 = convert_float_to_float16(model, disable_shape_infer=False, keep_io_types=True, op_block_list = op_block_list)
onnx.save(model_fp16, "../encoder_"+arch+"_fp16.opt3.onnx")

model = onnx.load("../decoder_"+arch+"_fix_kv_cache.opt3.onnx")
model_fp16 = convert_float_to_float16(model, disable_shape_infer=False, keep_io_types=True, op_block_list = op_block_list)
onnx.save(model_fp16, "../decoder_"+arch+"_fix_kv_cache_fp16_temp.opt3.onnx")

import onnx
from onnx import helper

def modify_onnx(model_path: str, output_path: str):
    model = onnx.load(model_path)
    graph = model.graph
    modified = False

    # モデル出力名のリスト
    output_names = [out.name for out in graph.output]

    # =========================================
    # Cast → output に複数接続されている場合の処理
    # =========================================
    for node in graph.node:
        if node.op_type != "Cast":
            continue

        cast_output = node.output[0]
        cast_input = node.input[0] if node.input else None
        if not cast_input:
            continue

        # このCastの出力がモデル出力に使われているか？
        connected_to_output = cast_output in output_names
        if not connected_to_output:
            continue  # 出力に直結していなければ対象外

        # Castの出力を使用している他ノードを探す
        for other in graph.node:
            if other == node:
                continue
            for i, inp_name in enumerate(other.input):
                if inp_name == cast_output:
                    # output専用にしたいので、output以外のノードはCastをスキップして直接入力に接続
                    other.input[i] = cast_input
                    print(f"Changed '{other.name}' input {i}: {cast_output} -> {cast_input}")
                    modified = True

    # =========================================
    # Cast が 2 回連続している場合、最初の Cast を削除
    # =========================================
    nodes_to_remove = []

    for node in list(graph.node):
        if node.op_type != "Cast":
            continue

        # 単一入力単一出力でなければスキップ
        if len(node.input) != 1 or len(node.output) != 1:
            continue

        cast_output = node.output[0]

        # 次の Cast ノードを探す
        next_cast = None
        for other in graph.node:
            if other.op_type == "Cast" and len(other.input) == 1 and len(other.output) == 1:
                if other.input[0] == cast_output:
                    next_cast = other
                    break

        if next_cast is not None:
            # 次の Cast の入力を最初の Cast の入力に変更
            print(f"Removing redundant Cast('{node.name}') before Cast('{next_cast.name}')")
            next_cast.input[0] = node.input[0]
            nodes_to_remove.append(node)
            modified = True

    # 削除予定の Cast ノードを取り除く
    for n in nodes_to_remove:
        graph.node.remove(n)

    # =========================================
    # 保存
    # =========================================
    if modified:
        onnx.save(model, output_path)
        print(f"Saved modified model to: {output_path}")
    else:
        print("No modifications were applied.")


modify_onnx("../decoder_"+arch+"_fix_kv_cache_fp16_temp.opt3.onnx", "../decoder_"+arch+"_fix_kv_cache_fp16.opt3.onnx")

import subprocess
subprocess.call("python3 onnx2prototxt.py " + "../encoder_"+arch+"_fp16.opt3.onnx", shell=True)
subprocess.call("python3 onnx2prototxt.py " + "../decoder_"+arch+"_fix_kv_cache_fp16.opt3.onnx", shell=True)