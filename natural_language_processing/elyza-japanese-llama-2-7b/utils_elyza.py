import numpy as np



def generate_prompt(tokenizer, DEFAULT_SYSTEM_PROMPT, text):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    
    prompt = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
            bos_token=tokenizer.bos_token,
            b_inst=B_INST,
            system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",
            prompt=text,
            e_inst=E_INST)
    
    return prompt

def generate_text(tokenizer, model, span, outputlength, onnx_runtime=False):
    
    #produce the initial tokens.
    encoding=tokenizer.encode_plus(span, return_tensors="np", add_special_tokens=False)
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    print("Tokenizer is done")
    
    # Initialize the generated tokens list
    generated_tokens = input_ids.copy()
    min_token_idx = -32000
    max_token_idx = 31999
    eos_token_id = tokenizer.eos_token_id 
    attention_mask = attention_mask.reshape(1, -1)

    for _ in range(outputlength):
        # Create input dictionary
        input_dict = {
            "input_ids": np.array(generated_tokens, dtype=np.int64),
            "attention_mask": np.array(attention_mask, dtype=np.int64)
        }

        # Run the inference to get logits
        if onnx_runtime:
            logits = model.run(None,input_dict)
        else:
            logits = model.run(input_dict)
        logits = np.array(logits[0])
        
        # Get the logits for the next token
        next_token_logits = logits[0, -1, :]

        # Sample the next token using the logits (you may use different strategies for sampling)
        next_token_id = np.argmax(next_token_logits)
        next_token_id = max(min_token_idx, min(max_token_idx, next_token_id))
        generated_tokens = np.concatenate((generated_tokens, [[next_token_id]]), axis=-1)

        # Update the attention_mask to consider the newly generated token
        attention_mask = np.concatenate((attention_mask, np.ones((1, 1), dtype=np.int64)), axis=1)
        
        if next_token_id == eos_token_id:
            break
            
    out_str = tokenizer.decode(generated_tokens[0][input_ids.shape[1]: ],  skip_special_tokens=True)
    

    return out_str 
    
   
