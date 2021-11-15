import numpy as np


def generate_text(tokenizer, ailia_model, span, outputlength, onnx_runtime=False):
    model_input = tokenizer.encode_plus(span)
    model_input = {name : np.atleast_2d(value) for name, value in model_input.items()}

    model_input['input_ids'] = np.array(model_input['input_ids'][:, :7], dtype='int64')
    model_input['attention_mask'] = np.array(model_input['attention_mask'][:, :7], dtype='int64')
    extra_inputs = np.array(model_input['input_ids'][:, 7:], dtype='int64')[0]

    if onnx_runtime:
      onnx_result = ailia_model.run(None,model_input)
    else:
      onnx_result = ailia_model.run(model_input)

    K=outputlength
    predictions = np.argpartition(-onnx_result[0][0, -1], K)[:K]

    out_str = span
    extra_len = extra_inputs.shape[0]
    for i in range(outputlength+extra_len):
      index = predictions[0]
      token = tokenizer.convert_ids_to_tokens([index])[0]
      if (extra_len-1) >= i:
          index = extra_inputs[i]
          token = tokenizer.convert_ids_to_tokens([index])[0]
      out_str += token.replace('Ġ',' ')
      input = np.append(model_input['input_ids'][:,1:], index)
      model_input['input_ids'] = np.expand_dims(input, 0)
      if onnx_runtime:
        out = ailia_model.run(None,model_input)
      else:
        out = ailia_model.run(model_input)
      predictions = np.argpartition(-out[0][0, -1], K)[:K]

      if token == "<unk>":
        break

    return out_str