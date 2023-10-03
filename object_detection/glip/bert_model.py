import numpy as np

language_dim = 768
num_layers = 1

onnx = False


def bert_encoder(net, input, mask, token):
    if not onnx:
        output = net.predict([input, mask, token])
    else:
        output = net.run(None, {
            'input_ids': input,
            'attention_mask': mask,
            'token_type_ids': token,
        })
    last_hidden_state, _ = output[:2]
    hidden_states = output[2:]

    # outputs has 13 layers, 1 input layer and 12 hidden layers
    encoded_layers = hidden_states[1:]

    features = np.stack(encoded_layers[-num_layers:], axis=1)
    features = np.mean(features, axis=1)

    # language embedding has shape [len(phrase), seq_len, language_dim]
    features = features / num_layers

    embedded = features * mask[:, :, None]
    aggregate = np.sum(embedded, axis=1) / np.sum(mask, axis=-1)[:, None]

    ret = {
        "aggregate": aggregate,
        "embedded": embedded,
        "masks": mask,
        # "hidden": encoded_layers[-1],
        "hidden": last_hidden_state
    }
    return ret
