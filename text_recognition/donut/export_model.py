import json
import sys
import argparse
from types import MethodType
from typing import Optional, Tuple, Union

import cv2
from PIL import Image
import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions


sys.path.insert(0, "./donut")
from donut import DonutModel


parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="cord-v2")
parser.add_argument(
    "--pretrained_path", type=str, default="naver-clova-ix/donut-base-finetuned-cord-v2"
)
args, left_argv = parser.parse_known_args()


def forward(
    self,
    hidden_states: torch.Tensor,
    key_value_states: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # if key_value_states are provided this layer is used as a cross-attention layer
    # for the decoder
    is_cross_attention = key_value_states is not None

    bsz, tgt_len, _ = hidden_states.size()

    # get query proj
    query_states = self.q_proj(hidden_states)
    # get key, value proj
    # `past_key_value[0].shape[2] == key_value_states.shape[1]`
    # is checking that the `sequence_length` of the `past_key_value` is the same as
    # the provided `key_value_states` to support prefix tuning
    if is_cross_attention:
        key_states = torch.cat(
            [past_key_value[0], self._shape(self.k_proj(key_value_states), -1, bsz)],
            dim=2,
        )
        value_states = torch.cat(
            [past_key_value[1], self._shape(self.v_proj(key_value_states), -1, bsz)],
            dim=2,
        )
        key_states = key_states[:, :, : key_value_states.shape[1], :]
        value_states = value_states[:, :, : key_value_states.shape[1], :]
    else:
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    if self.is_decoder:
        # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
        # Further calls to cross_attention layer can then reuse all cross-attention
        # key/value_states (first "if" case)
        # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
        # all previous decoder key/value_states. Further calls to uni-directional self-attention
        # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
        # if encoder bi-directional self-attention `past_key_value` is always `None`
        past_key_value = (key_states, value_states)

    query_states = self._shape(query_states, tgt_len, bsz)

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # The tgt_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case tgt_len == 1.
    is_causal = (
        True if self.is_causal and attention_mask is None and tgt_len > 1 else False
    )

    # NOTE: SDPA with memory-efficient backend is currently (torch==2.1.2) bugged when using non-contiguous inputs and a custom attn_mask,
    # but we are fine here as `_shape` do call `.contiguous()`. Reference: https://github.com/pytorch/pytorch/issues/112577
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2)

    # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
    # partitioned across GPUs when using tensor-parallelism.
    attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

    attn_output = self.out_proj(attn_output)

    return attn_output, None, past_key_value


def main():
    pretrained_model = DonutModel.from_pretrained(args.pretrained_path)
    pretrained_model.decoder.model.__class__.__name__ = "DonutModel"

    # DonutModel
    org_prepare_inputs_for_generation = (
        pretrained_model.decoder.model.prepare_inputs_for_generation
    )

    def prepare_inputs_for_inference(
        self,
        input_ids: torch.Tensor,
        encoder_outputs: torch.Tensor,
        past_key_values=None,
        past=None,
        use_cache: bool = None,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ):
        model_inputs = org_prepare_inputs_for_generation(
            input_ids,
            encoder_outputs,
            past_key_values=past_key_values,
            past=past,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )

        if model_inputs["past_key_values"] is None:
            model_inputs["past_key_values"] = [
                [
                    torch.zeros(1, 16, 0, 64, dtype=torch.float16).to(input_ids.device),
                ]
                * 4
            ] * 4

        return model_inputs

    pretrained_model.decoder.model.prepare_inputs_for_generation = MethodType(
        prepare_inputs_for_inference, pretrained_model.decoder.model
    )

    # MBartSdpaAttention
    for decoder_layer in pretrained_model.decoder.model.model.decoder.layers:
        decoder_layer.self_attn.forward = MethodType(forward, decoder_layer.self_attn)
        decoder_layer.encoder_attn.forward = MethodType(
            forward, decoder_layer.encoder_attn
        )

    task_name = args.task
    task_prompt = f"<s_{task_name}>"

    img = cv2.imread("cord_sample_receipt1.png")[..., ::-1]
    img = Image.fromarray(img)
    output = pretrained_model.inference(image=img, prompt=task_prompt)["predictions"][0]

    print(json.dumps(output, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
