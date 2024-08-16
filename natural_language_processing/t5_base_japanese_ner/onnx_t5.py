import numpy as np


def softmax_np(x: np.ndarray):
    e_x = np.exp(x - np.max(x))  # subtract max to stabilize
    return e_x / e_x.sum(axis=0)



"""
top k top p filtering algorithm
Modified by Takumi Ibayashi.
"""
def top_k_top_p_filtering(logits: np.ndarray, top_k:int=0, top_p:float=0.0, filter_value:float=-float("Inf")):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (vocabulary size)
        top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert (
        logits.ndim == 1
    )  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.shape[-1])  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < np.partition(logits, -top_k)[-top_k]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_indices = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]
        cumulative_probs = np.cumsum(np.exp(sorted_logits) / np.sum(np.exp(sorted_logits)))

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1]
        sorted_indices_to_remove[0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

"""
model wrapper
"""
class T5Model:
    """
    This class is based on `GenerativeT5` from `models` in `onnxt5`.
    Modified by Takumi Ibayashi.
    """
    def __init__(self, encoder, decoder_with_lm_head, tokenizer):
        super().__init__()
        self.encoder = encoder
        self.decoder_with_lm_head = decoder_with_lm_head
        self.tokenizer = tokenizer

    def estimate(
        self, prompt: str, max_length: int, temperature:float=1.0, repetition_penalty:float=1.0, top_k:int=50, top_p:int=0, max_context_length: int=512, 
    ):
        """
        Generate a text output given a prompt using the model.

        Args:
            prompt (str): The initial text input to the model, which it uses as a 
                starting point to generate the subsequent text.
            max_length (int): The maximum length of the text to be generated.
            temperature (float, optional): This controls the randomness in the model's 
                text generation. A higher temperature value results in more random output. 
                If the temperature is very small, it will approach greedy decoding. 
                Defaults to 1.0.
            top_k (int, optional): parameter for top k filtering algorithm
            top_p (int, optional): parameter for top p filtering algorithm
            repetition_penalty (float, optional): This increases the model's likelihood 
                to generate diverse output by discouraging it from repeating the same 
                token. Defaults to 1.0.
            max_context_length (int, optional): The maximum length of the context to be 
                used in generation. Defaults to 512.
        """
        new_tokens = np.array([], dtype=np.float32)
        new_logits = []

        # generate tokens with tokenizer
        enc = self.tokenizer.encode_plus(#encode tokens
            text=prompt,
            max_length=512,
            truncation=True,
        )

        enc_input =(
            np.array(enc['input_ids'])[None,:],#prepare input
            np.array(enc['attention_mask'])[None,:],
        )

        # encode tokens
        encoder_outputs_prompt = self.encoder.run(enc_input)[0]

        # reset token
        token = np.zeros((1,1), dtype=np.int64)
        for _ in range(max_length):
            # decode tokens
            outputs = np.array(
                self.decoder_with_lm_head.run(
                    (token, encoder_outputs_prompt, np.ones_like(token)),
                )[0][0]
            )
            next_token_logits = outputs[-1, :] / (temperature if temperature > 0 else 1.0)

            # `1` means end of sentence. (EOS token)
            if int(next_token_logits.argmax()) in (1,0):
                break

            new_logits.append(next_token_logits)
            for _ in set(token.reshape(-1).tolist()):
                next_token_logits[_] /= repetition_penalty

            # select next token
            if temperature == 0:
                # greedy sampling: this methods always choose the most probable token.
                next_token = np.expand_dims(np.argmax(next_token_logits), axis=0)
            else:
                # Top-k and top-p filtering: this methods enhance text diversity and creativity.
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                probs = softmax_np(filtered_logits)
                next_token = np.expand_dims(np.random.choice(np.arange(len(probs)), p=probs), axis=0)
            token = np.concatenate((token, np.expand_dims(next_token, axis=0)), axis=1)
            new_tokens = np.concatenate((new_tokens, next_token), axis=0)
        return new_tokens.astype('int'), new_logits