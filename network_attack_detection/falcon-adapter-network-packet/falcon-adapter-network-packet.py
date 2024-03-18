import sys
import time
import joblib
from logging import getLogger

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from transformers import AutoTokenizer
from scapy.all import Ether, CookedLinux

import ailia

# import original modules
sys.path.append("../../util")
from arg_utils import get_base_parser, update_parser
from model_utils import check_and_download_models, check_and_download_file

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_ENC_PATH = "reazonspeech-nemo-v2_encoder.onnx"
MODEL_ENC_PATH = "reazonspeech-nemo-v2_encoder.onnx.prototxt"
REMOTE_PATH = (
    "https://storage.googleapis.com/ailia-models/falcon-adapter-network-packet/"
)

SAVE_TEXT_PATH = "output.txt"

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    "falcon-adapter-network-packet",
    None,
    SAVE_TEXT_PATH,
)
parser.add_argument("--onnx", action="store_true", help="execute onnxruntime version.")
args = update_parser(parser)


# ======================
# Secondaty Functions
# ======================


# ======================
# Main functions
# ======================


def predict(models, packet_hex):
    packet_bytes = bytes.fromhex(packet_hex)
    packet = Ether(packet_bytes)
    if packet.firstlayer().name != "Ethernet":
        packet = CookedLinux(packet_bytes)
        if packet.firstlayer().name != "cooked linux":
            raise ValueError(
                f"{packet.firstlayer().name} frame not implemented. Ethernet and Cooked Linux are only supported."
            )

    forward_packets_per_second = 0
    backward_packets_per_second = 4
    bytes_transferred_per_second = 5493

    source_port = packet["TCP"].sport
    destination_port = packet["TCP"].dport
    IP_len = packet["IP"].len
    IP_ttl = packet["IP"].ttl
    IP_tos = f"0x{str(packet['IP'].tos)}"
    tos_map = {
        "0x0": 0,
        "0x10": 1,
        "0x18": 2,
        "0x2": 3,
        "0x20": 4,
        "0x28": 5,
        "0x34": 6,
        "0x4": 7,
        "0x40": 8,
        "0x48": 9,
        "0x60": 10,
        "0x68": 11,
        "0x8": 12,
        "0x88": 13,
    }
    IP_tos = tos_map.get(IP_tos, 14)
    TCP_dataofs = packet["TCP"].dataofs
    TCP_flags = str(packet["TCP"].flags)
    flags_map = {"A": 0, "FA": 1, "FPA": 2, "PA": 3}
    TCP_flags = flags_map.get(TCP_flags, 4)
    if packet.haslayer("Raw"):
        payload_hex = packet.load.hex()
        payload_len = len(payload_hex) // 2
    else:
        raise ValueError(
            "Network Packet does not contain a payload. This model is trained with a payload."
        )
    payload_hex = packet.load.hex()
    payload_hex = payload_hex

    payload_len = len(payload_hex) // 2
    payload = [int(payload_hex[i : i + 2], 16) for i in range(0, len(payload_hex), 2)]

    final_format = [
        forward_packets_per_second,
        backward_packets_per_second,
        bytes_transferred_per_second,
        -1,
        source_port,
        destination_port,
        IP_len,
        payload_len,
        IP_ttl,
        IP_tos,
        TCP_dataofs,
        TCP_flags,
        -1,
    ]
    final_format = final_format + payload[:500]
    final_format = [str(i) for i in final_format]
    final_format = " ".join(final_format)

    tokenizer = models["tokenizer"]
    similar = models["similar"]
    tokens = tokenizer(final_format, truncation=True, return_tensors="np")
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    # feedforward
    if not args.onnx:
        output = similar.predict([input_ids, attention_mask])
    else:
        output = similar.run(
            None, {"input_ids": input_ids, "attention_mask": attention_mask}
        )
    last_hidden_state = output[0]

    embedding = np.mean(last_hidden_state[:, 1:-1, :], axis=1)

    scaler = models["scaler"]
    df = pd.DataFrame(embedding[0].reshape(1, -1), columns=[str(i) for i in range(768)])
    embedding = scaler.transform(df)[0]

    cluster_centers = models["cluster_centers"]

    euclidean_distance = euclidean_distances(cluster_centers, [embedding])
    data = {i: euclidean_distance[i][0] for i in range(len(euclidean_distance))}
    df = pd.DataFrame(list(data.items()), columns=["Cluster", "Euclidean Distance"])
    df.sort_values(by="Euclidean Distance", inplace=True)

    cluster, distance, embedding = (
        int(df.iloc[0]["Cluster"]),
        df.iloc[0]["Euclidean Distance"],
        embedding,
    )

    return 0


def recognize_from_packet(models):
    # input audio loop
    args.input = [""]
    for packet_path in args.input:
        logger.info(packet_path)

        # prepare input data
        packet_hex = "3ca6f60849b920b39957e74b0800450005c881dc0000f506c2790d235d2b86588b3301bbf95a94eccbfa554bbac980100085d54400000101080abcb794b10c6ab7722057d82613cc2c721b879ef00e6d925bca92a02d529fd587fd8e5a9cb93dd2a405d8315612500d7179cf7c01ca5e18cd137fe2044fe15898d5b42722f9e79bbc7431ce711171aa63a6b779367d745a0b5432fa326e8e7238d15033da601a4bb9c9bea464f6ca54b64698f31493d9da42fa6e0904a15fb1f944b96de8c55909f7e8780be2de10786b0ff623e503f94276a694bbf823686654ebcdafbfce9f5677e3d21ac1d25426a2be1badeadc5f29449a024419bba4d350ce7494563e9dabaa2c405e21a5fc918586193499139bd967d06ad188e8446ce0ddd406a336847bb64e1e70a73aaffdd1fdfc8cddd89b73433fe0fdcfe11dffa208710e0ecec840b632071872bb688353f59740f45d1efec153e2cc2b69f756b871073a8af9ca923eb213df7c1a67f5679d64e3e758394695fa486c32fd43d454bacc5b5f733eb5e28f70d605ff0947cf68e27dd51081b08ee083976d6b6eb277bd5e8787cb80e0bd574b6f6493e626999467e098ec329fd049d7d20ddc18547e2284e5560509692ce6e86fee5ece2997757697279dbbe418c37a86a79829b34cf8cb52e07e389c61373eff20705d8906aa6d98d5169bb316e963c6a85c8a4f5aea12d6e9a5402cb2aacc63be2b5a845bb5be1f416e19764f44b57837a854d233b764cbb8849f49a5c3deb77a0208cb512d973034c36d90870efdbad00c55fc3d85ef76fd275c21cf0cfbd6cf3cebbd0c62d3c4e8cb21a65b0983c1ed24d9f0a2bd1831316d62aeb6ec9e14a998803671b12d4dcf37151b75b69ec28cca72a36f67b5d3ec3f02606f94ebf941c0f705fd3ba39a154dcb20b1929df10c2ced9db7de3f2bfca59528e699591436b605ae5c174e3c3d7a237c72a0cce22d4cc370767d78a7ed485eb5fc96f6ae45e7e3114ecb1aab59acdcc14a7303b4f49484c2b834f8289e006bd4c6ae38018db9c48ea09caa095b25a0e626486713e07ca409ff52918d6bd390903db3b3a5f823cb91dab2d515c34f459c58dd242529322bc10428786451bd7c2d899f0398c9ffc37302b0d2dca95569d29db478705ed7c85a27ec00cb827c4671424ee33a49a80ec1e63b3a810af84ea42bdac72b6c9a5aa5438bdc4461a9bf3dafc676457072918c6c6a65aaed79a1be272f006edf7c2e930919a53a2eae0749d98cdd9c1b482d4db4adb7a9865ac613bb9a9d8110a72f3f4f40a58fe9fa8eec36e1eee61124d84e92001c617fb025e48e250a173e031552575b48e67d67c988c432364e945e5b3845d61090ccbb628504aac0d453a91c75fa23d6d59b65eadfe79c10f9878715780b9c5b68df37234ddd723b0023611c647f17fddaf0266eec2faa7e745fb06017cbcba1608fd3a9903036d3c5505a3185d0b31f512106509a4cc5582fe13283a18d817b95feb25a61782f2a571722c24979fb39efaf823be465483271e4c4dcc39a8cbc930492ed1b224aa37c50dc19e67b4f1117f92d0bd6ef81cbc72ac2189e27d893b838a19d7a2b8a9b46a6786fdbcfa3749cf564b0038440418a7c9fe2f477458ef743270aeafe0bf510f043a7e7d54787ab92ba80f97d75e06f4bc25cb521d54d221fd089d408d7c9166268376c5c2de1c2f44dc6c0402c35a0f55b2f3ea13f80a11a80f65d41bcb63dac7ae9cfa063a8c749231d6d2cd9b5a83252972f0dd424efa79b72bf558d1648dd2c78c202e7398eef6b8adeab334227e92534e7f3dd26bdaa856ce1feba77f87005e4ed87a6dae4c2bb2c72eecfaaf9e1299cb2f0ff1f3f8cff459e30396bf595d7c08a9a704a394211cc459e01a939cb6cbf8627ceefebb1b338d47079e3958009d2388b86e38a9a5c51f2134c304f98c21d00951c8aa15d3f47e9ba61fa43606d91698000bb7427365ef8b485d11bcdfcea0d52e40af2b76e9f3d372b15c9463b18660f23cd5f04e660f727467a34d8994b22f713f1bfaaf2cb1a0b2aaaa3b1caacd6955ec3e96fde2ca82b5caedc45521cb3978a7c3d65b4076ec96f069608"

        # inference
        logger.info("Start inference...")
        if args.benchmark:
            logger.info("BENCHMARK mode")
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                output = predict(models, packet_hex)
                end = int(round(time.time() * 1000))
                estimation_time = end - start

                # Logging
                logger.info(f"\tailia processing estimation time {estimation_time} ms")
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(
                f"\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms"
            )
        else:
            output = predict(models, packet_hex)

    logger.info("Script finished successfully.")


def main():
    # model files check and download
    # check_and_download_models(WEIGHT_ENC_PATH, MODEL_ENC_PATH, REMOTE_PATH)

    env_id = args.env_id

    # initialize
    if not args.onnx:
        # encoder = ailia.Net(MODEL_ENC_PATH, WEIGHT_ENC_PATH, env_id=env_id)
        # decoder = ailia.Net(MODEL_DEC_PATH, WEIGHT_DEC_PATH, env_id=env_id)
        # joint = ailia.Net(MODEL_JNT_PATH, WEIGHT_JNT_PATH, env_id=env_id)
        pass
    else:
        import onnxruntime

        # encoder = onnxruntime.InferenceSession(WEIGHT_ENC_PATH)
        # decoder = onnxruntime.InferenceSession(WEIGHT_DEC_PATH)
        # joint = onnxruntime.InferenceSession(WEIGHT_JNT_PATH)
        similar = onnxruntime.InferenceSession("similar.onnx")

    tokenizer = AutoTokenizer.from_pretrained("tokenizer")

    scaler = joblib.load("scaler.pkl")
    cluster_centers = np.load("KMEANS-CLUSTER-CENTERS.npy")

    models = {
        "tokenizer": tokenizer,
        "similar": similar,
        "scaler": scaler,
        "cluster_centers": cluster_centers,
        # "joint": joint,
    }

    recognize_from_packet(models)


if __name__ == "__main__":
    main()
