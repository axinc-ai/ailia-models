import cv2
import numpy as np
from scipy.special import softmax

from constants import CONVERSATION_START, STOP_STR, EOS_TOKEN_ID
from preprocessing_numpy import llamaMultimodalInputProcesor, ImageProcessor, simple_prompt_preprocessor_single_image, tokenize_w_image_token, simple_prompt_preprocessor


def predict(model, input, onnx:bool = False):
    if onnx:
        out = model.run(None, input)
    else:
        out = model.predict(input)
    return out

def top_k_top_p_filtering(probabilities, top_k=0, top_p=0.0):
    if len(probabilities) >= top_k:
        topk_tok = np.argpartition(-probabilities, top_k)[: top_k]
    else:
        topk_tok = np.argsort(-probabilities)
    topk_prob = probabilities[topk_tok]

    topk_ind_sorted = np.argsort(topk_prob)[::-1]
    topk_tok_sorted = topk_tok[topk_ind_sorted]
    topk_prob_sorted = topk_prob[topk_ind_sorted]

    cumulative_probs = np.cumsum(topk_prob_sorted)
    cutoff = np.searchsorted(cumulative_probs, top_p, side='right')+1
    topp_ind_sorted = topk_tok_sorted[:cutoff]
    topp_prob_sorted = topk_prob_sorted[:cutoff]
    topp_prob_sorted /= np.sum(topp_prob_sorted)
    return topp_ind_sorted, topp_prob_sorted

def sample_next_token(logits, temperature, top_k, top_p):
    logits = logits / temperature
    logits = softmax(logits)
    ind, proba = top_k_top_p_filtering(logits, top_k, top_p)
    next_token = np.random.choice(ind, p=proba)
    return next_token

def generate_text(
        tokenizer, models, prompt, image_paths,
        max_len: int = 512, temperature: float = 0.0, top_p: float = 0.9, top_k: int = 10,
        onnx: bool = False, use_kvcache: bool = True):
    
    prompt = simple_prompt_preprocessor_single_image(prompt)
    #prompt = simple_prompt_preprocessor(prompt)
    print('processed prompt: \n', prompt)
    with open('prompt.txt', 'w') as f:
        f.write(prompt)
    
    input_ids = tokenize_w_image_token(prompt, tokenizer)
    print(input_ids)
    images = [cv2.imread(p) for p in image_paths]
    imp = ImageProcessor(
        convert_to_rgb = True,
        pad_to_square = True,
        image_size = (336, 336),
        rescale_factor = 1/255,
        resample = cv2.INTER_AREA,
        as_channel_first = True,
        add_batch_dimension = False
    )
    images = np.stack([imp(im) for im in images])
    multimodalprocessor = llamaMultimodalInputProcesor(
        models['vision_tower'],
        models['mm_projector'],
        models['token_embedder']
    )
    input_ids = input_ids
    input_embeds, attention_mask = multimodalprocessor([input_ids], images)
    output_tokens = []

    kvcache = np.load('kvcache_bos.npy')
    cur_len = input_embeds.shape[1]
    #input_embeds = input_embeds[:, 1:]
    while True:
        if len(output_tokens) != 0:
            new_token_embeds = predict(models['token_embedder'],
                np.array([output_tokens[-1]])
            , onnx=onnx)[None]
            if use_kvcache:
                input_embeds = new_token_embeds
            else:
                input_embeds = np.concatenate([input_embeds, new_token_embeds], axis=1)
        print(input_embeds[:, :, 0])

        print(input_embeds.shape, attention_mask.shape, kvcache.shape)

        #logits, kvcache = predict(models['mobilellama'], {
        #logits, = predict(models['mobilellama'], {
        #    "input_embeds": input_embeds.astype(np.float16),
        #    "attention_mask": attention_mask.astype(np.int64),
        #    #"past_kvs": kvcache
        #}, onnx=onnx)
        with torch.inference_mode():
            out = model.model(
                inputs_embeds = torch.tensor(input_embeds).cuda(),
                attention_mask = torch.tensor(attention_mask).cuda()
            )
            logits = model.lm_head(out['last_hidden_state']).cpu().numpy()
        logits = logits[:, -1]

        if not use_kvcache:
            kvcache = np.load('kvcache_bos.npy')
        
        cur_len += 1
        
        if temperature == 0.0:
            print(logits.shape)
            next_token = logits.argmax()
        else:
            next_token = sample_next_token(logits[0], temperature, top_k, top_p)
        output_tokens.append(next_token)
        attention_mask = np.concatenate([attention_mask, np.ones((1, 1))], axis=1)

        print(next_token)
        if next_token == EOS_TOKEN_ID or cur_len >= max_len:
            break
        #break
    print('output_tokens: \n', output_tokens)
    return tokenizer.decode(output_tokens, skip_special_tokens=True)
