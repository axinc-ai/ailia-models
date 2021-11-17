import numpy as np


def generate_text(tokenizer, ailia_model, span, outputlength, onnx_runtime=False):
    model_input = tokenizer.encode_plus(span)
    model_input = {name : np.atleast_2d(value) for name, value in model_input.items()}

    model_input['input_ids'] = np.array(model_input['input_ids'], dtype='int64')
    model_input['attention_mask'] = np.array(model_input['attention_mask'], dtype='int64')

    if onnx_runtime:
      onnx_result = ailia_model.run(None,model_input)
    else:
      onnx_result = ailia_model.run(model_input)

    out_str = span
    for i in range(outputlength):
      K=outputlength
      predictions = np.argpartition(-onnx_result[0][0, -1], K)[:K]
      index = predictions[0]
      token = tokenizer.convert_ids_to_tokens([index])[0]
      out_str += token.replace('Ġ',' ')
      trim = 0
      input = np.append(model_input['input_ids'][:,trim:], index)
      model_input['input_ids'] = np.expand_dims(input, 0)
      attention_mask = np.append(model_input['attention_mask'][:,trim:], 1)
      model_input['attention_mask'] = np.expand_dims(attention_mask, 0)
      if onnx_runtime:
        onnx_result = ailia_model.run(None,model_input)
      else:
        onnx_result = ailia_model.run(model_input)

      if token == "<unk>":
        break

    return out_str