import json

import torch
import torch.nn as nn
from torch.nn import functional as F
import transformers
from transformers import BertModel, BertPreTrainedModel


class ModifiedFocalLoss(nn.Module):
    def __init__(self, alpha=0, gamma=0.7, reduction='mean', eps=1e-6):
        super().__init__()
        if reduction not in ['mean', 'none', 'sum']:
            raise NotImplementedError('Reduction {} not implemented.'.format(reduction))
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, probs, target):
        probs = torch.clamp(probs, min=self.eps, max=1-self.eps)
        target = F.one_hot(target, num_classes=probs.size(1))
        p_t = torch.where(target == 1, probs, 1 - probs)
        losses = - 1 * (1 + self.alpha - p_t) ** self.gamma * torch.log(p_t)
        return self._reduce(losses)

    def _reduce(self, losses):
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses


class G2PW(BertPreTrainedModel):
    def __init__(self, model_source, labels, chars, pos_tags,
                 use_conditional=False, param_conditional=None,
                 use_focal=False, param_focal=None,
                 use_pos=False, param_pos=None):
        super().__init__(model_source)

        self.num_labels = len(labels)
        self.num_chars = len(chars)
        self.num_pos_tags = len(pos_tags)

        self.bert = BertModel(self.config)

        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)

        self.use_conditional = use_conditional
        self.param_conditional = param_conditional
        if self.use_conditional:
            conditional_affect_location = self.param_conditional['affect_location']
            target_size = self.config.hidden_size if conditional_affect_location == 'emb' else self.num_labels

            if self.param_conditional['bias']:
                self.descriptor_bias = nn.Embedding(1, target_size)
            if self.param_conditional['char-linear']:
                self.char_descriptor = nn.Embedding(self.num_chars, target_size)
            if self.param_conditional['pos-linear']:
                self.pos_descriptor = nn.Embedding(self.num_pos_tags, target_size)
            if self.param_conditional['char+pos-second']:
                self.second_order_descriptor = nn.Embedding(self.num_chars * self.num_pos_tags, target_size)
            if self.param_conditional['char+pos-second_lowrank']:
                assert not self.param_conditional['char+pos-second']
                assert 0 < self.param_conditional['lowrank_size'] < target_size
                self.second_lowrank_descriptor = nn.Sequential(
                    nn.Embedding(self.num_chars * self.num_pos_tags, self.param_conditional['lowrank_size']),
                    nn.Linear(self.param_conditional['lowrank_size'], target_size)
                )
            if self.param_conditional['char+pos-second_fm']:
                assert not self.param_conditional['char+pos-second']
                assert 0 < self.param_conditional['fm_size']
                self.second_fm_char_emb = nn.Sequential(
                    nn.Embedding(self.num_chars, self.param_conditional['fm_size'] * target_size),
                    nn.Unflatten(1, (target_size, self.param_conditional['fm_size']))
                )
                self.second_fm_pos_emb = nn.Sequential(
                    nn.Embedding(self.num_pos_tags, self.param_conditional['fm_size'] * target_size),
                    nn.Unflatten(1, (target_size, self.param_conditional['fm_size']))
                )
            if self.param_conditional['fix_mode']:
                assert all([not self.param_conditional[x] for x in ['bias', 'char-linear', 'pos-linear', 'char+pos-second', 'char+pos-second_lowrank', 'char+pos-second_fm']])
                assert self.param_conditional['affect_location'] == 'softmax'
                count_dict = json.load(open(self.param_conditional['count_json']))
                if self.param_conditional['fix_mode'] == 'count_distr:char':
                    char_fix_count = torch.tensor(
                        [[count_dict['by_char'][char].get(label, 0.) for label in labels] for char in chars]
                    )
                    self.char_fix_emb = nn.parameter.Parameter(
                        char_fix_count / char_fix_count.sum(dim=-1, keepdim=True),
                        requires_grad=False)
                elif self.param_conditional['fix_mode'] == 'count_distr:char+pos':
                    char_pos_fix_count = torch.tensor(
                        [[count_dict['by_char_pos'][f'{char}-{pos}'].get(label, 0.)
                          if f'{char}-{pos}' in count_dict['by_char_pos'] else 0.
                          for label in labels]
                         for char in chars for pos in pos_tags]
                    )
                    self.char_pos_fix_emb = nn.parameter.Parameter(
                        char_pos_fix_count / char_pos_fix_count.sum(dim=-1, keepdim=True),
                        requires_grad=False)
                else:
                    raise Exception

        self.use_focal = use_focal
        self.param_focal = param_focal

        self.use_pos = use_pos
        self.param_pos = param_pos
        if self.use_pos and self.param_pos['pos_joint_training']:
            self.pos_classifier = nn.Linear(self.config.hidden_size, self.num_pos_tags)

    def _weighted_softmax(self, logits, weights, eps):
        max_logits, _ = torch.max(logits, dim=-1, keepdim=True)
        weighted_exp_logits = torch.exp(logits - max_logits) * weights
        norm = torch.sum(weighted_exp_logits, dim=-1, keepdim=True)
        probs = weighted_exp_logits / norm
        probs = torch.clamp(probs, min=eps, max=1-eps)
        return probs

    def _get_char_pos_ids(self, char_ids, pos_ids):
        return char_ids * self.num_pos_tags + pos_ids

    def _get_pos_loss_scaling_when_using_focal(self, phoneme_probs, label_ids):
        phoneme_probs = phoneme_probs.detach()
        phoneme_target = F.one_hot(label_ids, num_classes=phoneme_probs.size(1))
        phoneme_p_t = torch.where(phoneme_target == 1, phoneme_probs, 1 - phoneme_probs)
        avg_phoneme_p_t = phoneme_p_t.mean()
        scaling = (1 + self.param_focal['alpha'] - avg_phoneme_p_t) ** self.param_focal['gamma']
        return scaling

    def forward(self, input_ids, token_type_ids, attention_mask, phoneme_mask, char_ids, position_ids, pos_ids=None, label_ids=None, eps=1e-6):
        transformers_major_ver = int(transformers.__version__.split('.')[0])
        if transformers_major_ver >= 4:
            sequence_output, pooled_output = self.bert(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                return_dict=False
            )
        else:
            sequence_output, pooled_output = self.bert(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask
            )

        batch_size = input_ids.size(0)
        orig_selected_hidden = sequence_output[torch.arange(batch_size), position_ids]
        selected_hidden = orig_selected_hidden

        if self.use_conditional:
            if (self.param_conditional['char+pos-second']
                    or self.param_conditional['char+pos-second_lowrank']
                    or self.param_conditional['char+pos-second_fm']
                    or self.param_conditional['pos-linear']
                    or self.param_conditional['fix_mode'] == 'count_distr:char+pos'):
                pred_pos_ids = pos_ids if self.training or not self.param_pos['pos_joint_training'] \
                    else self.pos_classifier(orig_selected_hidden).argmax(dim=-1)  # teacher mode while training

            affect_terms = []
            if self.param_conditional['bias']:
                bias_tensor = self.descriptor_bias(torch.zeros_like(char_ids))
                affect_terms.append(bias_tensor)
            if self.param_conditional['char-linear']:
                affect_terms.append(self.char_descriptor(char_ids))
            if self.param_conditional['pos-linear']:
                affect_terms.append(self.pos_descriptor(pred_pos_ids))
            if self.param_conditional['char+pos-second']:
                char_pos_ids = self._get_char_pos_ids(char_ids, pred_pos_ids)
                affect_terms.append(self.second_order_descriptor(char_pos_ids))
            if self.param_conditional['char+pos-second_lowrank']:
                char_pos_ids = self._get_char_pos_ids(char_ids, pred_pos_ids)
                affect_terms.append(self.second_lowrank_descriptor(char_pos_ids))
            if self.param_conditional['char+pos-second_fm']:
                affect_terms.append(
                    torch.sum(
                        self.second_fm_char_emb(char_ids) * self.second_fm_pos_emb(pred_pos_ids),
                        dim=-1
                    )
                )
            affect_hidden = sum(affect_terms)

            if self.param_conditional['fix_mode'] == 'count_distr:char':
                phoneme_mask = phoneme_mask * F.embedding(char_ids, self.char_fix_emb)
            elif self.param_conditional['fix_mode'] == 'count_distr:char+pos':
                char_pos_ids = self._get_char_pos_ids(char_ids, pred_pos_ids)
                phoneme_mask = phoneme_mask * F.embedding(char_pos_ids, self.char_pos_fix_emb)
            elif self.param_conditional['affect_location'] == 'emb':
                selected_hidden = selected_hidden * affect_hidden
            elif self.param_conditional['affect_location'] == 'softmax':
                phoneme_mask = phoneme_mask * torch.sigmoid(affect_hidden)
            else:
                raise Exception

        logits = self.classifier(selected_hidden)
        probs = self._weighted_softmax(logits, phoneme_mask, eps)
        if label_ids is not None:
            if self.use_focal:
                loss_layer = ModifiedFocalLoss(alpha=self.param_focal['alpha'], gamma=self.param_focal['gamma'])
                loss = loss_layer(probs, label_ids)
            else:
                loss_layer = nn.NLLLoss()
                log_probs = torch.log(probs)
                loss = loss_layer(log_probs, label_ids)

            pos_logits = None
            if self.use_pos and pos_ids is not None and self.param_pos['pos_joint_training']:
                pos_logits = self.pos_classifier(orig_selected_hidden)
                loss_fct = nn.CrossEntropyLoss()
                pos_loss = loss_fct(pos_logits, pos_ids)
                scaling = self._get_pos_loss_scaling_when_using_focal(probs, label_ids) if self.use_focal else 1.
                loss += self.param_pos['weight'] * scaling * pos_loss

            return probs, loss, pos_logits
        else:
            return probs
