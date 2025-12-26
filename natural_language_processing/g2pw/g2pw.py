import sys
import os
from logging import getLogger

import ailia

# g2pwライブラリのコンポーネントを直接インポート
from g2pw.api import G2PWConverter

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser  # noqa
from model_utils import check_and_download_models  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
WEIGHT_PATH = "G2PWModel/g2pw.onnx"
MODEL_PATH = "G2PWModel/g2pw.onnx" # .prototxtは不要なためonnxファイルと同じパスを指定
REMOTE_PATH = ""

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    "G2PW",
    None,
    None,
)
parser.add_argument(
    "-i",
    "--input",
    type=str,
    default="你好世界",
    help="Input text.",
)
parser.add_argument(
    "--style",
    type=str,
    default="bopomofo",
    choices=["bopomofo", "pinyin"],
    help="Output style. (bopomofo or pinyin)"
)
args = update_parser(parser, check_input_type=False)


class AiliaG2P(G2PWConverter):
    """
    元のリポジトリのapi部分をオーバーライド
    """
    def __init__(self, model_path, weight_path, env_id, style='bopomofo', **kwargs):
        # G2PWConverterはmodel_dirを要求するため、weight_pathからディレクトリを取得
        model_dir = os.path.dirname(weight_path) or '.'
        # 親クラスの初期化を呼び出す。ONNXランタイムのセッションは作らせない
        self.session_g2pw = None
        super().__init__(model_dir=model_dir, style=style, use_cuda=False, **kwargs)

        # ailiaモデルを初期化
        logger.info("Initializing ailia model...")
        self.net = ailia.Net(None, weight_path, env_id=env_id)
        logger.info("ailia model initialized.")
        def ailia_run_wrapper(session_self, outputs, inputs):
            return self.net.predict(inputs)

        # G2PWConverter内の self.session_g2pw.run を ailia_run_wrapper で置き換える
        self.session_g2pw = type('session', (object,), {})()
        self.session_g2pw.run = ailia_run_wrapper.__get__(self, type(self))


def main():
    # model files check and download
    # check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)
    converter = AiliaG2P(
        model_path=MODEL_PATH,
        weight_path=WEIGHT_PATH,
        env_id=args.env_id,
        style=args.style,
        num_workers=0
    )

    texts_to_infer = [args.input]

    # 親クラスの __call__ メソッド (内部で predict が呼ばれる) を実行
    results = converter(texts_to_infer)

    # 結果を表示
    logger.info("--- Input ----")
    logger.info(args.input)
    logger.info(f"--- Output : {args.style} ---")
    # resultsは二次元配列 [[...]] なので、そのまま文字列に変換して表示
    logger.info(str(results))

    logger.info("Script finished successfully.")


if __name__ == '__main__':
    main()